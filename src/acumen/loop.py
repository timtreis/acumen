"""The autoresearch loop — crude P6+P7 prototype: optimize the rulebook, score on held-out.

**This is a deliberately crude prototype**, built to answer one feasibility question before we
invest in P2–P5: *does optimizing the rulebook (the SKILL-generating instructions) — rather than a
skill directly — actually move a held-out score, and is the failure signal legible enough to drive
the next rulebook version?* If yes, the real loop (CV over tasks, difficulty strata, coverage, the
selection-leakage lockbox) is worth building. If the signal is mush, we redesign first.

What it does, one iteration:

1. Seed rulebook ``v1`` from the built-in draft prompt (today's drafting behaviour, verbatim).
2. Draft a skill from ``v1``; benchmark it on the tasks (both splits, on a Claude subscription).
3. Score the rulebook = the skill's **test-split** pass rate (the held-out variant the skill never
   had evidence from).
4. Run the outer-improve agent: it reads ``v1`` + the **train-split** failure evidence and writes
   rulebook ``v2`` — editing the instructions, never the instance.
5. Draft a skill from ``v2``; benchmark on the test split; score again. Report ``v1`` vs ``v2``.

**What is crude here, and honest about it.** "Held-out" is the existing *within-task* test split
(``tasks.py`` train/test variants), not a partition of tasks — so leakage protection comes free from
the same ``runs/*/test/`` guard the skill improver uses, and no new slicing is needed. The real
CV-over-tasks axis, difficulty strata, coverage-driven generation, warm cache, and the nested-CV
lockbox are all still ahead (P2–P5). One improvement step only; one model, whatever ``n_replicates``
the config says. The rulebook artifact itself (:mod:`acumen.rulebooks`) is a versioned file, not the
content-hashed, meta-tracked artifact the real P6 will be.

Because the inner scoring runs on a subscription, the loop calls the benchmark machinery with
``auth_mode="session"`` directly — it scores pass rate, not dollars, so it does not need the
API-metered billing the ``acumen bench`` command forces.
"""

from __future__ import annotations

import difflib
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from acumen import rulebooks as rb
from acumen.bench import PlannedRun, build_matrix, pending, run_matrix
from acumen.config import Config
from acumen.draft import draft_skill
from acumen.env import AuthMode, Target, build_agent_env
from acumen.improve import (
    _read_rationale,
    _write_material,
    collect_train_runs,
    make_test_guard,
)
from acumen.logs import LiveLog
from acumen.paths import RESULT_FILE, SPLITS, Split, arm_name, is_complete, run_dir
from acumen.procs import label_env, reap
from acumen.prompts import rulebook_improve_prompt
from acumen.runner import RunOutcome
from acumen.skills import Skill, available_versions, load_skill
from acumen.tasks import Task


class LoopError(RuntimeError):
    """Raised when a loop iteration cannot proceed."""


@dataclass(frozen=True)
class RulebookResult:
    """A newly written rulebook version and what it cost to produce."""

    version: str
    parent: str
    path: Path
    rationale: str
    changed: bool
    cost_usd: float
    turns: int
    n_train_runs: int
    n_train_failures: int
    log_jsonl: Path | None = None
    log_html: Path | None = None


@dataclass(frozen=True)
class Score:
    """Two independent success metrics over a benchmarked slice: passing, and loading the skill.

    ``passed`` and ``loaded`` are orthogonal (a baseline run can pass without any skill; a skill run
    can load the skill and still fail), so they are tracked separately — a skill that never loads is
    a distinct, and prior, failure to one that loads and gets the wrong answer. The first real
    measured loop run turned entirely on this distinction: every arm failed *because the skill never
    loaded*, invisible on pass rate alone.
    """

    passed: int
    total: int
    #: How many of these runs actually loaded the skill under test (0 for the noskill arm).
    loaded: int = 0

    @property
    def rate(self) -> float:
        """Pass rate in ``[0, 1]``; ``0.0`` when nothing was scored."""
        return self.passed / self.total if self.total else 0.0

    @property
    def load_rate(self) -> float:
        """Skill-load rate in ``[0, 1]``; ``0.0`` when nothing was scored."""
        return self.loaded / self.total if self.total else 0.0


@dataclass(frozen=True)
class LoopResult:
    """The outcome of one crude v1->v2 rulebook iteration."""

    baseline_version: str
    improved_version: str
    baseline_skill: str
    improved_skill: str
    #: Test-split (held-out) pass rate for each rulebook version.
    baseline_score: Score
    improved_score: Score
    #: Train-split pass rate for rulebook v1, for context on what drove the improvement.
    baseline_train_score: Score
    #: Held-out pass rate with NO skill (the floor a skill must beat), read from the noskill arm if
    #: it was benched in ``runs_root`` (e.g. by a prior ``acumen screen``); ``total==0`` if absent.
    #: This is the "does the skill help at all" signal, distinct from the "did the rulebook improve"
    #: signal (baseline_score -> improved_score).
    noskill_score: Score
    rulebook: RulebookResult
    rulebook_diff: str
    cost_usd: float

    @property
    def moved(self) -> int:
        """Change in held-out passes from rulebook v1 to v2 (can be negative)."""
        return self.improved_score.passed - self.baseline_score.passed

    @property
    def load_moved(self) -> int:
        """Change in held-out skill-load count from rulebook v1 to v2 (can be negative).

        A rulebook edit aimed at the ``description`` moves loading, not passing — and a skill that
        does not load cannot pass at all, so this is the earlier, often larger, signal of progress.
        """
        return self.improved_score.loaded - self.baseline_score.loaded


# ── Scoring ────────────────────────────────────────────────────────────────────────────


def score(runs_root: Path, planned: Sequence[PlannedRun]) -> Score:
    """Read ``result.json`` for each planned run and tally both success metrics: passes and loads.

    Reads from disk rather than trusting a ``run_matrix`` return, so a resumed pass (where the
    matrix ran nothing because everything was already complete) still scores correctly.
    """
    passed = total = loaded = 0
    for item in planned:
        directory = run_dir(runs_root, item.key)
        if not is_complete(directory):
            continue
        try:
            data = json.loads((directory / RESULT_FILE).read_text())
        except (OSError, ValueError):
            continue
        total += 1
        passed += bool(data.get("success"))
        loaded += bool(data.get("skill_loaded"))
    return Score(passed=passed, total=total, loaded=loaded)


# ── Benchmarking a skill (subscription auth, pass rate not dollars) ─────────────────────


async def _bench(
    *,
    cfg: Config,
    target: Target,
    skill: Skill,
    tasks: Sequence[Task],
    runs_root: Path,
    splits: Sequence[Split],
    auth_mode: AuthMode,
    task_ids: Sequence[str] | None,
    max_concurrency: int,
    on_start: Callable[[PlannedRun], None] | None,
    on_done: Callable[[RunOutcome], None] | None,
) -> list[PlannedRun]:
    """Bench one skill on the given splits, resuming completed runs. Returns the full planned set.

    The loop scores from disk (:func:`score`) over the returned planned set, so this ignores the
    outcomes ``run_matrix`` hands back.
    """
    planned = build_matrix(cfg, tasks, skill=skill.version, splits=splits, task_ids=task_ids)
    todo = pending(planned, runs_root, resume=True)
    if todo:
        await run_matrix(
            todo,
            target=target,
            runs_root=runs_root,
            max_concurrency=max_concurrency,
            auth_mode=auth_mode,
            skill=skill,
            skill_name=cfg.skill_name,
            on_start=on_start,
            on_done=on_done,
            env_passthrough=cfg.env_passthrough,
            dataset_cache_dirs=cfg.dataset_cache_dirs,
        )
    return planned


# ── The outer-improve agent (edits the rulebook, not the skill) ─────────────────────────


async def improve_rulebook(
    *,
    cfg: Config,
    target: Target,
    rulebooks_root: Path,
    runs_root: Path,
    benched_skill: Skill,
    tasks: Sequence[Task],
    auth_mode: AuthMode = "session",
    parent_version: str | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    feedback: str | None = None,
    log: LiveLog | None = None,
) -> RulebookResult:
    """Improve the current rulebook into the next version from the drafted skill's train evidence.

    The outer-loop analogue of :func:`acumen.improve.improve_skill`: same train-split evidence, same
    held-out guard, but the artifact under edit is the rulebook (draft instructions), not the skill.
    ``benched_skill`` is the skill drafted from the parent rulebook and just benchmarked — its arm is
    where the train evidence lives.

    Raises
    ------
    LoopError
        If there is no parent rulebook, no train evidence, or the agent produced an invalid rulebook.
    """
    parent = parent_version or rb.latest_version(rulebooks_root)
    if parent is None:
        raise LoopError(f"no rulebook to improve under {rulebooks_root} — seed one first")
    parent_text = rb.load_rulebook(rulebooks_root, parent)
    new_version = rb.next_version(rulebooks_root)
    if rb.rulebook_dir(rulebooks_root, new_version).exists():
        raise LoopError(f"rulebook {new_version} already exists — versions are immutable")

    arm = arm_name(benched_skill.version)
    train_runs = collect_train_runs(runs_root, arm, list(tasks))
    if not train_runs:
        raise LoopError(
            f"no train-split runs for {benched_skill.version} under {runs_root / arm / 'train'} — "
            "the skill must be benched on the train split before the rulebook can be improved"
        )
    n_failures = sum(1 for r in train_runs if not r.success)

    holder = Path(tempfile.mkdtemp(prefix="acumen-rulebook-"))
    try:
        work = holder / "work"
        staged_rulebook = work / rb.RULEBOOK_FILE
        train_dir = work / "train"
        rationale_path = work / "rationale.md"
        home = holder / "home"
        config_dir = home / ".claude"
        for path in (work, home, config_dir, home / "tmp"):
            path.mkdir(parents=True, exist_ok=True)

        staged_rulebook.write_text(parent_text)
        # Reuse the skill improver's evidence writer verbatim — the material is identical.
        _write_material(train_dir, train_runs)

        env = label_env(
            build_agent_env(
                config_dir=config_dir,
                home=home,
                extra_path=[target.bin_dir],
                auth_mode=auth_mode,
                extra_allow=cfg.env_passthrough,
            ),
            holder,
        )
        prompt = rulebook_improve_prompt(
            package=target.pkg_name,
            rulebook_path=staged_rulebook,
            train_dir=train_dir,
            rationale_path=rationale_path,
            parent_version=parent,
            new_version=new_version,
            feedback=feedback,
        )
        options = ClaudeAgentOptions(
            cwd=str(work),
            env=env,
            model=model or cfg.improve_model,
            max_turns=max_turns,
            max_budget_usd=max_usd,
            setting_sources=["project"],
            permission_mode="bypassPermissions",
            system_prompt={"type": "preset", "preset": "claude_code"},
            # Same held-out guard as the skill improver: the rulebook agent must not see test runs.
            hooks={"PreToolUse": [make_test_guard(runs_root)]},
        )

        result: ResultMessage | None = None
        agent_error: Exception | None = None
        try:
            async for message in query(prompt=prompt, options=options):
                if log is not None:
                    log.append(message)
                if isinstance(message, ResultMessage):
                    result = message
        except Exception as err:  # noqa: BLE001 - a failed improve is an error to report, re-raised below
            agent_error = err
        finally:
            if log is not None:
                log.finalize(config_dir=config_dir, work_dir=work, result=result)

        if agent_error is not None:
            raise LoopError(f"the rulebook-improve agent failed: {type(agent_error).__name__}: {agent_error}") from (
                agent_error
            )
        if result is None:
            raise LoopError("the rulebook-improve agent produced no result message")
        if result.is_error:
            raise LoopError(f"the rulebook-improve agent errored: {result.subtype} {result.errors or ''}".strip())

        new_text = staged_rulebook.read_text()
        try:
            # write_rulebook validates placeholders — a broken template fails here, not at draft time.
            rb.write_rulebook(rulebooks_root, new_version, new_text)
        except rb.RulebookError as err:
            raise LoopError(f"the rulebook-improve agent produced an invalid rulebook: {err}") from err

        rationale = _read_rationale(rationale_path, result, parent, new_version)
        (rb.rulebook_dir(rulebooks_root, new_version) / "rationale.md").write_text(rationale + "\n")
        return RulebookResult(
            version=new_version,
            parent=parent,
            path=rb.rulebook_dir(rulebooks_root, new_version) / rb.RULEBOOK_FILE,
            rationale=rationale,
            changed=new_text != parent_text,
            cost_usd=result.total_cost_usd or 0.0,
            turns=result.num_turns,
            n_train_runs=len(train_runs),
            n_train_failures=n_failures,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)


# ── One crude iteration ─────────────────────────────────────────────────────────────────


async def _ensure_skill(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebook_text: str,
    expect_version: str,
    auth_mode: AuthMode,
    rationale: str,
    log_dir: Path | None,
    stream: bool,
) -> tuple[Skill, float]:
    """Draft a skill from ``rulebook_text`` if ``expect_version`` isn't present, else load it.

    Resume-friendly: a re-run reuses an already-drafted skill instead of erroring on the immutable
    version. Asserts the drafted version matches ``expect_version`` so the rulebook<->skill lockstep
    the loop relies on can't silently drift.
    """
    if expect_version in available_versions(skills_root):
        return load_skill(skills_root, expect_version, expect_name=cfg.skill_name), 0.0
    log = LiveLog.open(log_dir, f"loop-draft-{expect_version}", stream=stream) if log_dir is not None else None
    try:
        result = await draft_skill(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            auth_mode=auth_mode,
            rationale=rationale,
            rulebook=rulebook_text,
            log=log,
        )
    finally:
        if log is not None:
            log.close()
    if result.skill.version != expect_version:
        raise LoopError(
            f"drafted skill {result.skill.version} but expected {expect_version} — the "
            "rulebook/skill version lockstep is broken; run the loop in a clean workspace"
        )
    return result.skill, result.cost_usd


async def _ensure_rulebook(
    *,
    cfg: Config,
    target: Target,
    rulebooks_root: Path,
    runs_root: Path,
    benched_skill: Skill,
    tasks: Sequence[Task],
    baseline_version: str,
    baseline_text: str,
    auth_mode: AuthMode,
    feedback: str | None,
    log_dir: Path | None,
    stream: bool,
) -> RulebookResult:
    """Improve the rulebook, or — on a resumed run — reconstruct the already-improved version.

    Keeps the loop resumable: if a version past the baseline already exists on disk, the (expensive)
    improve agent is not re-run; its recorded rationale is read back so the report is still complete.
    """
    latest = rb.latest_version(rulebooks_root)
    if latest is not None and latest != baseline_version:
        text = rb.load_rulebook(rulebooks_root, latest)
        rationale_file = rb.rulebook_dir(rulebooks_root, latest) / "rationale.md"
        rationale = (
            rationale_file.read_text().strip() if rationale_file.is_file() else "(resumed; rationale not recorded)"
        )
        return RulebookResult(
            version=latest,
            parent=baseline_version,
            path=rb.rulebook_dir(rulebooks_root, latest) / rb.RULEBOOK_FILE,
            rationale=rationale,
            changed=text != baseline_text,
            cost_usd=0.0,
            turns=0,
            n_train_runs=0,
            n_train_failures=0,
        )
    log = LiveLog.open(log_dir, "loop-rulebook-v2", stream=stream) if log_dir is not None else None
    try:
        return await improve_rulebook(
            cfg=cfg,
            target=target,
            rulebooks_root=rulebooks_root,
            runs_root=runs_root,
            benched_skill=benched_skill,
            tasks=tasks,
            auth_mode=auth_mode,
            parent_version=baseline_version,
            feedback=feedback,
            log=log,
        )
    finally:
        if log is not None:
            log.close()


async def run_iteration(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    runs_root: Path,
    tasks: Sequence[Task],
    auth_mode: AuthMode = "session",
    max_concurrency: int | None = None,
    task_ids: Sequence[str] | None = None,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    on_bench_start: Callable[[PlannedRun], None] | None = None,
    on_bench_done: Callable[[RunOutcome], None] | None = None,
) -> LoopResult:
    """Run one crude v1->v2 rulebook iteration and report whether the held-out score moved.

    See the module docstring for what is deliberately crude. The steps: seed rulebook v1 -> draft
    skill v1 -> bench both splits -> score test split -> improve rulebook to v2 from train evidence
    -> draft skill v2 -> bench test split -> score. Resumable: drafting and benching both skip work
    already on disk, so an interrupted run continues.
    """
    concurrency = max_concurrency or cfg.max_concurrency
    total_cost = 0.0

    baseline_version = rb.seed_default(rulebooks_root)
    if baseline_version != "v1":
        raise LoopError(
            f"expected to start from rulebook v1 but found {baseline_version} — run the loop in a "
            "clean workspace (empty rulebooks/, skills/, runs/)"
        )
    baseline_text = rb.load_rulebook(rulebooks_root, baseline_version)

    # v1: draft, bench both splits, score.
    skill_v1, cost = await _ensure_skill(
        cfg=cfg,
        target=target,
        skills_root=skills_root,
        rulebook_text=baseline_text,
        expect_version="v1",
        auth_mode=auth_mode,
        rationale=f"drafted from rulebook {baseline_version}",
        log_dir=log_dir,
        stream=stream,
    )
    total_cost += cost
    planned_v1 = await _bench(
        cfg=cfg,
        target=target,
        skill=skill_v1,
        tasks=tasks,
        runs_root=runs_root,
        splits=SPLITS,
        auth_mode=auth_mode,
        task_ids=task_ids,
        max_concurrency=concurrency,
        on_start=on_bench_start,
        on_done=on_bench_done,
    )
    baseline_score = score(runs_root, [p for p in planned_v1 if p.key.split == "test"])
    baseline_train_score = score(runs_root, [p for p in planned_v1 if p.key.split == "train"])

    # Outer improvement: rulebook v1 -> v2 from the train evidence (skipped on a resumed run
    # where the improved version already exists).
    rulebook = await _ensure_rulebook(
        cfg=cfg,
        target=target,
        rulebooks_root=rulebooks_root,
        runs_root=runs_root,
        benched_skill=skill_v1,
        tasks=tasks,
        baseline_version=baseline_version,
        baseline_text=baseline_text,
        auth_mode=auth_mode,
        feedback=feedback,
        log_dir=log_dir,
        stream=stream,
    )
    total_cost += rulebook.cost_usd
    improved_text = rb.load_rulebook(rulebooks_root, rulebook.version)

    # v2: draft from the improved rulebook, bench the test split, score.
    skill_v2, cost = await _ensure_skill(
        cfg=cfg,
        target=target,
        skills_root=skills_root,
        rulebook_text=improved_text,
        expect_version="v2",
        auth_mode=auth_mode,
        rationale=f"drafted from rulebook {rulebook.version}",
        log_dir=log_dir,
        stream=stream,
    )
    total_cost += cost
    planned_v2 = await _bench(
        cfg=cfg,
        target=target,
        skill=skill_v2,
        tasks=tasks,
        runs_root=runs_root,
        splits=["test"],
        auth_mode=auth_mode,
        task_ids=task_ids,
        max_concurrency=concurrency,
        on_start=on_bench_start,
        on_done=on_bench_done,
    )
    improved_score = score(runs_root, [p for p in planned_v2 if p.key.split == "test"])

    # The floor: how the no-skill baseline did on the same held-out, if it was benched into this
    # runs tree (e.g. by `acumen screen`). Read-only — the loop does not run it, so a run without a
    # prior baseline simply reports total==0. This is the "does the skill help at all" comparison.
    noskill_score = score(runs_root, build_matrix(cfg, tasks, skill=None, splits=["test"], task_ids=task_ids))

    diff = "".join(
        difflib.unified_diff(
            baseline_text.splitlines(keepends=True),
            improved_text.splitlines(keepends=True),
            fromfile=f"rulebook {baseline_version}",
            tofile=f"rulebook {rulebook.version}",
        )
    )
    return LoopResult(
        baseline_version=baseline_version,
        improved_version=rulebook.version,
        baseline_skill=skill_v1.version,
        improved_skill=skill_v2.version,
        baseline_score=baseline_score,
        improved_score=improved_score,
        baseline_train_score=baseline_train_score,
        noskill_score=noskill_score,
        rulebook=rulebook,
        rulebook_diff=diff,
        cost_usd=total_cost,
    )
