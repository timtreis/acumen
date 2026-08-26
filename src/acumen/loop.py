"""The autoresearch loop: optimize the *rulebook*, score it on analyses it never saw.

The rulebook (:mod:`acumen.rulebooks`) is the SKILL-generating instructions; skills are drafted from
it and are intermediates. One improvement step reads a drafted skill's **train-split** failure
evidence and rewrites the rulebook — the instructions, never the instance — then a fresh skill is
drafted from the new version and benchmarked. Three entry points, in order of honesty:

* :func:`run_iteration` — the original single ``v1 -> v2`` step, scored on the tasks' own **test
  variants**. A *within-task* signal: it says whether a skill memorised an answer, not whether the
  rulebook generalises. Kept as the quick smoke test it is.
* :func:`run_cv_iteration` — the same step **cross-validated over tasks** (:mod:`acumen.folds`): per
  fold the rulebook is improved from the optimize tasks' evidence only, behind a guard that denies the
  held-out tasks' runs in every split, and scored on the held-out tasks. The mean held-out delta is
  the estimate; the refit on all working tasks is the version carried forward.
* :func:`run_loop` — iterates that until a :class:`StopRule` fires, picks the version with the best
  cross-validated rate, and then — once — benches the pick and the seed on the **lockbox**, the task
  set nothing in the loop ever read. That delta is the loop's one honest generalisation number.

Every boundary is structural: evidence is collected only for the tasks an agent may learn from, and a
``PreToolUse`` guard (:func:`acumen.improve.make_test_guard`) denies everything else — test splits,
held-out tasks, the lockbox, the CV and lockbox run trees. Everything resumes by file presence, with
each iteration's parent/carried versions pinned so a resumed loop replays the same chain.

Optional task selection by measured difficulty (``headroom_only``, :mod:`acumen.difficulty`) narrows
the working set to tasks the no-skill baseline does not already solve, before any agent runs.

Scoring runs on a Claude subscription (``auth_mode="session"``): the loop optimizes pass rate and
skill-load rate, never dollars.
"""

from __future__ import annotations

import difflib
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from acumen import rulebooks as rb
from acumen.bench import PlannedRun, build_matrix, pending, run_matrix
from acumen.config import Config
from acumen.difficulty import HeadroomSelection, screen, select_headroom
from acumen.draft import draft_skill
from acumen.env import AuthMode, Target, build_agent_env
from acumen.folds import Fold, FoldError, Lockbox, check_disjoint, load_lockbox_tasks, make_folds, read_lockbox
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
from acumen.skills import Skill, available_versions, load_skill, version_number
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
    #: How the task set was chosen when ``headroom_only`` was on: what was kept and what was left
    #: out (baseline-solved, or never screened). ``None`` when every given task was used as-is.
    selection: HeadroomSelection | None = None

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
    held_out_ids: Sequence[str] = (),
    deny_dirs: Sequence[Path] = (),
) -> RulebookResult:
    """Improve the current rulebook into the next version from the drafted skill's train evidence.

    The outer-loop analogue of :func:`acumen.improve.improve_skill`: same train-split evidence, same
    held-out guard, but the artifact under edit is the rulebook (draft instructions), not the skill.
    ``benched_skill`` is the skill drafted from the parent rulebook and just benchmarked — its arm is
    where the train evidence lives.

    The evidence is structurally limited to ``tasks``: only their train runs are collected and
    written into the agent's work dir. For a CV fold, ``tasks`` is the optimize set and
    ``held_out_ids`` the rest — the guard then also denies the held-out tasks' runs in *every*
    split (their train runs name the task and its answer), and ``deny_dirs`` (the lockbox, the CV
    trees) wholesale. Evidence and guard together are the fold boundary.

    Raises
    ------
    LoopError
        If there is no parent rulebook, no train evidence, or the agent produced an invalid rulebook.
    """
    parent = parent_version or rb.latest_version(rulebooks_root)
    if parent is None:
        raise LoopError(f"no rulebook to improve under {rulebooks_root} — seed one first")
    parent_text = rb.load_rulebook(rulebooks_root, parent).text
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
            # Same held-out guard as the skill improver — plus, for a CV fold, the held-out tasks
            # and the lockbox/CV trees. See make_test_guard.
            hooks={"PreToolUse": [make_test_guard(runs_root, held_out_ids=held_out_ids, deny_dirs=deny_dirs)]},
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
        rationale = _read_rationale(rationale_path, result, parent, new_version)
        try:
            # write_rulebook validates placeholders — a broken template fails here, not at draft
            # time — and records parent/rationale/hash in meta.json, the provenance chain.
            written = rb.write_rulebook(
                rulebooks_root, new_version, new_text, parent=parent, rationale=rationale, feedback=feedback
            )
        except rb.RulebookError as err:
            raise LoopError(f"the rulebook-improve agent produced an invalid rulebook: {err}") from err

        return RulebookResult(
            version=new_version,
            parent=parent,
            path=written.path,
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
    deny_dirs: Sequence[Path] = (),
    log_name: str = "loop-rulebook-v2",
    expect_version: str | None = None,
) -> RulebookResult:
    """Improve the rulebook, or — on a resumed run — reconstruct the already-improved version.

    Keeps the loop resumable: if the improved version already exists on disk, the (expensive)
    improve agent is not re-run; its recorded rationale is read back so the report is still complete.
    ``expect_version`` pins which version counts as "the improved one" — a multi-iteration loop
    resuming iteration 1 of a three-version chain must reconstruct ``v2``, not the latest ``v3``.
    Without it, the latest version past the baseline is taken (the single-iteration behaviour).
    """
    latest = expect_version if expect_version in rb.available_versions(rulebooks_root) else None
    if expect_version is None:
        latest = rb.latest_version(rulebooks_root)
        if latest == baseline_version:
            latest = None
    elif latest is None and rb.next_version(rulebooks_root) != expect_version:
        raise LoopError(
            f"expected to write rulebook {expect_version} but the chain's next version is "
            f"{rb.next_version(rulebooks_root)} — the rulebook chain is not linear; run in a clean workspace"
        )
    if latest is not None:
        resumed = rb.load_rulebook(rulebooks_root, latest)
        meta = rb.rulebook_meta(rulebooks_root, latest)
        rationale = meta.rationale if meta is not None and meta.rationale else "(resumed; rationale not recorded)"
        return RulebookResult(
            version=latest,
            parent=meta.parent if meta is not None and meta.parent else baseline_version,
            path=resumed.path,
            rationale=rationale,
            changed=resumed.text != baseline_text,
            cost_usd=0.0,
            turns=0,
            n_train_runs=0,
            n_train_failures=0,
        )
    log = LiveLog.open(log_dir, log_name, stream=stream) if log_dir is not None else None
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
            deny_dirs=deny_dirs,
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
    headroom_only: bool = False,
    on_select: Callable[[HeadroomSelection], None] | None = None,
    on_bench_start: Callable[[PlannedRun], None] | None = None,
    on_bench_done: Callable[[RunOutcome], None] | None = None,
) -> LoopResult:
    """Run one crude v1->v2 rulebook iteration and report whether the held-out score moved.

    See the module docstring for what is deliberately crude. The steps: seed rulebook v1 -> draft
    skill v1 -> bench both splits -> score test split -> improve rulebook to v2 from train evidence
    -> draft skill v2 -> bench test split -> score. Resumable: drafting and benching both skip work
    already on disk, so an interrupted run continues.

    With ``headroom_only``, the task set is first narrowed to the tasks the no-skill baseline does
    not already max out on the held-out split, judged per reference model against ``cfg.models``
    (:func:`acumen.difficulty.select_headroom`). Tasks never screened are excluded, not guessed at:
    scoring a rulebook on a task the baseline aces cannot show movement, so it is wasted agent
    time and, worse, dilutes the signal. ``on_select`` sees the decision before any agent runs.

    Raises
    ------
    LoopError
        If ``headroom_only`` leaves no task to score on.
    """
    concurrency = max_concurrency or cfg.max_concurrency
    total_cost = 0.0

    selection: HeadroomSelection | None = None
    if headroom_only:
        diffs = screen(runs_root, tasks, by_model=True)
        selection = select_headroom(diffs, tasks, split="test", models=cfg.models)
        if task_ids:
            wanted = set(task_ids)
            selection = HeadroomSelection(
                selected=[t for t in selection.selected if t.id in wanted],
                solved=selection.solved,
                unscreened=selection.unscreened,
            )
        if on_select is not None:
            on_select(selection)
        if not selection.selected:
            raise LoopError(
                "no task has headroom for the loop: the baseline already passes every screened task "
                f"({len(selection.solved)} solved) and {len(selection.unscreened)} were never screened — "
                "run `acumen bench --no-skill` on more tasks, or generate harder ones"
            )
        tasks = selection.selected
        task_ids = None

    baseline_version = rb.seed_default(rulebooks_root)
    if baseline_version != "v1":
        raise LoopError(
            f"expected to start from rulebook v1 but found {baseline_version} — run the loop in a "
            "clean workspace (empty rulebooks/, skills/, runs/)"
        )
    baseline_text = rb.load_rulebook(rulebooks_root, baseline_version).text

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
    improved_text = rb.load_rulebook(rulebooks_root, rulebook.version).text

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
        selection=selection,
    )


# ── Cross-validated iteration (P5) ──────────────────────────────────────────────────────

#: Subdirectory (under each of rulebooks/, skills/, runs/) holding per-fold artifacts. Kept apart
#: from the linear ``vN`` chain: a fold's rulebook is an *estimate* of what the improvement
#: procedure buys on unseen analyses, not a version anything is carried forward from.
CV_DIRNAME = "cv"

#: Subdirectory under runs/ holding the lockbox evaluations — denied to every improve agent, since a
#: later iteration must not learn anything from how an earlier version did on the lockbox.
LOCKBOX_RUNS_DIRNAME = "lockbox"


@dataclass(frozen=True)
class FoldResult:
    """One fold's estimate: improve on the optimize tasks, score on the tasks never seen."""

    fold: Fold
    #: Where the fold's rulebook / skill / runs live (each its own root, so the linear chains
    #: and the main run tree are untouched).
    rulebooks_root: Path
    skills_root: Path
    runs_root: Path
    #: The main-tree skill (drafted from the parent rulebook) on this fold's held-out tasks.
    baseline_held_out: Score
    #: The skill drafted from the fold-improved rulebook on the same held-out tasks.
    improved_held_out: Score
    cost_usd: float

    @property
    def delta_rate(self) -> float:
        """Held-out pass-rate change the fold's improvement bought on analyses it never saw."""
        return self.improved_held_out.rate - self.baseline_held_out.rate

    @property
    def delta_load_rate(self) -> float:
        """Held-out skill-load-rate change, the earlier signal (a skill that never loads cannot pass)."""
        return self.improved_held_out.load_rate - self.baseline_held_out.load_rate


@dataclass(frozen=True)
class CVResult:
    """One cross-validated rulebook iteration: k fold estimates plus the refit that is carried."""

    baseline_version: str
    baseline_skill: str
    folds: list[FoldResult]
    #: The rulebook improved on ALL working tasks — the version carried forward (``vN+1``). Its own
    #: test-split score is within-task and therefore optimistic; the CV numbers are the estimate.
    carried: RulebookResult
    carried_skill: str
    baseline_test: Score
    carried_test: Score
    noskill_score: Score
    rulebook_diff: str
    cost_usd: float
    lockbox: Lockbox | None = None
    selection: HeadroomSelection | None = None

    @property
    def cv_deltas(self) -> list[float]:
        """Per-fold held-out pass-rate deltas."""
        return [f.delta_rate for f in self.folds]

    @property
    def cv_mean_delta(self) -> float:
        """Mean held-out pass-rate delta over folds — the cross-validated estimate of the gain."""
        return sum(self.cv_deltas) / len(self.folds) if self.folds else 0.0

    @property
    def cv_spread(self) -> float:
        """Max minus min fold delta: how much the estimate depends on which tasks were held out."""
        return (max(self.cv_deltas) - min(self.cv_deltas)) if self.folds else 0.0

    @property
    def cv_mean_load_delta(self) -> float:
        """Mean held-out skill-load-rate delta over folds."""
        return sum(f.delta_load_rate for f in self.folds) / len(self.folds) if self.folds else 0.0


def _subset(planned: Sequence[PlannedRun], task_ids: Sequence[str], split: Split) -> list[PlannedRun]:
    wanted = set(task_ids)
    return [p for p in planned if p.key.split == split and p.key.task_id in wanted]


async def run_cv_iteration(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    runs_root: Path,
    tasks: Sequence[Task],
    k: int = 3,
    seed: int = 0,
    lockbox_dir: Path | None = None,
    allow_no_lockbox: bool = False,
    parent_version: str | None = None,
    auth_mode: AuthMode = "session",
    max_concurrency: int | None = None,
    task_ids: Sequence[str] | None = None,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    headroom_only: bool = False,
    on_select: Callable[[HeadroomSelection], None] | None = None,
    on_fold: Callable[[FoldResult], None] | None = None,
    on_bench_start: Callable[[PlannedRun], None] | None = None,
    on_bench_done: Callable[[RunOutcome], None] | None = None,
) -> CVResult:
    """One rulebook iteration scored by cross-validation over tasks, behind a lockbox.

    The honest form of :func:`run_iteration`. That one scores a rulebook on the *test variants* of
    the very tasks whose train runs improved it — a within-task signal that says whether a skill
    memorised an answer, not whether the rulebook generalises. Here the working tasks are split
    into ``k`` folds (:func:`acumen.folds.make_folds`) and, per fold, the rulebook is improved from
    the **optimize** tasks' evidence only and scored on the **held-out** tasks — analyses the
    improve agent never saw. The boundary is structural twice over: the evidence written into the
    agent's work dir is collected for the optimize tasks alone, and a ``PreToolUse`` guard denies
    the held-out tasks' runs in every split, the lockbox, and every CV tree. The mean held-out delta
    over folds is the cross-validated estimate of what one improvement buys.

    The version **carried forward** is not a fold's rulebook but the refit: improved on all working
    tasks (the standard CV recipe — estimate the procedure out of sample, then fit on everything).
    Its own within-task test score is reported too, labelled as optimistic.

    A **lockbox** (:mod:`acumen.folds`) is required unless ``allow_no_lockbox``: every selection the
    loop makes is on CV scores, so those are optimistic by construction; the lockbox is the set no
    selection ever touched, scored once at the end by the outer loop. The working set is checked to
    be disjoint from it, and its directory is denied to every improve agent.

    Everything is resumable by file presence: shared v1 draft and bench, each fold's rulebook v2 /
    skill v1 / held-out runs, and the refit.

    Raises
    ------
    LoopError
        If the lockbox is missing (and not explicitly waived) or overlaps the working set, the fold
        request is degenerate, or ``headroom_only`` leaves nothing to score on.
    """
    concurrency = max_concurrency or cfg.max_concurrency
    total_cost = 0.0

    selection: HeadroomSelection | None = None
    if headroom_only:
        diffs = screen(runs_root, tasks, by_model=True)
        selection = select_headroom(diffs, tasks, split="test", models=cfg.models)
        if on_select is not None:
            on_select(selection)
        if not selection.selected:
            raise LoopError("no task has headroom for the loop — run `acumen bench --no-skill` on more tasks first")
        tasks = selection.selected
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.id in wanted]

    lockbox: Lockbox | None = None
    if lockbox_dir is not None:
        try:
            lockbox = read_lockbox(lockbox_dir)
            check_disjoint(tasks, lockbox)
        except FoldError as err:
            raise LoopError(str(err)) from err
    elif not allow_no_lockbox:
        raise LoopError(
            "no lockbox given — every score the loop selects on is optimistic, and the lockbox is the one "
            "that is not. Run `acumen lockbox` first and pass --lockbox, or pass --no-lockbox to proceed "
            "without a final held-out set (the result then cannot be trusted as a generalisation claim)"
        )
    try:
        folds = make_folds([t.id for t in tasks], k, seed)
    except FoldError as err:
        raise LoopError(str(err)) from err
    by_id = {t.id: t for t in tasks}

    # Shared: the parent rulebook, the skill drafted from it, and its bench on every working task.
    baseline_version = rb.seed_default(rulebooks_root)
    parent_version = parent_version or rb.latest_version(rulebooks_root) or baseline_version
    parent_text = rb.load_rulebook(rulebooks_root, parent_version).text
    # The carried version is pinned to parent+1 so a resumed multi-iteration loop replays each
    # iteration against the same versions it wrote the first time.
    carried_version = f"v{version_number(parent_version) + 1}"
    if (
        carried_version not in rb.available_versions(rulebooks_root)
        and rb.next_version(rulebooks_root) != carried_version
    ):
        raise LoopError(
            f"rulebook {parent_version} is not the head of a linear chain (next would be "
            f"{rb.next_version(rulebooks_root)}, iteration wants {carried_version}); run in a clean workspace"
        )
    parent_skill_version = f"v{version_number(parent_version)}"
    skill_parent, cost = await _ensure_skill(
        cfg=cfg,
        target=target,
        skills_root=skills_root,
        rulebook_text=parent_text,
        expect_version=parent_skill_version,
        auth_mode=auth_mode,
        rationale=f"drafted from rulebook {parent_version}",
        log_dir=log_dir,
        stream=stream,
    )
    total_cost += cost
    planned_parent = await _bench(
        cfg=cfg,
        target=target,
        skill=skill_parent,
        tasks=tasks,
        runs_root=runs_root,
        splits=SPLITS,
        auth_mode=auth_mode,
        task_ids=None,
        max_concurrency=concurrency,
        on_start=on_bench_start,
        on_done=on_bench_done,
    )
    baseline_test = score(runs_root, [p for p in planned_parent if p.key.split == "test"])

    # Denied to every improve agent this iteration: the lockbox and all CV trees (a fold must not
    # read another fold's held-out runs, which may be its own optimize tasks' answers).
    cv_root = runs_root / CV_DIRNAME
    deny_dirs: list[Path] = [cv_root, runs_root / LOCKBOX_RUNS_DIRNAME]
    if lockbox is not None:
        deny_dirs.append(lockbox.directory)

    fold_results: list[FoldResult] = []
    for fold in folds:
        tag = f"{carried_version}/fold-{fold.index}"
        fold_rb_root = rulebooks_root / CV_DIRNAME / carried_version / f"fold-{fold.index}"
        fold_sk_root = skills_root / CV_DIRNAME / carried_version / f"fold-{fold.index}"
        fold_runs_root = cv_root / carried_version / f"fold-{fold.index}"
        optimize_tasks = [by_id[t] for t in fold.optimize]
        held_out_tasks = [by_id[t] for t in fold.held_out]
        fold_cost = 0.0

        # The fold's own rulebook chain starts from a copy of the parent, so the fold's improve
        # writes v2 there and the linear main chain is untouched.
        if "v1" not in rb.available_versions(fold_rb_root):
            rb.write_rulebook(
                fold_rb_root,
                "v1",
                parent_text,
                parent=parent_version,
                rationale=f"copy of rulebook {parent_version} — the parent of fold {fold.index}'s estimate",
            )
        if rb.latest_version(fold_rb_root) == "v1":
            log = LiveLog.open(log_dir, f"loop-rulebook-{tag}", stream=stream) if log_dir is not None else None
            try:
                improved = await improve_rulebook(
                    cfg=cfg,
                    target=target,
                    rulebooks_root=fold_rb_root,
                    runs_root=runs_root,
                    benched_skill=skill_parent,
                    tasks=optimize_tasks,
                    auth_mode=auth_mode,
                    parent_version="v1",
                    feedback=feedback,
                    log=log,
                    held_out_ids=fold.held_out,
                    deny_dirs=deny_dirs,
                )
            finally:
                if log is not None:
                    log.close()
            fold_cost += improved.cost_usd
        fold_text = rb.load_rulebook(fold_rb_root, "v2").text

        fold_skill, cost = await _ensure_skill(
            cfg=cfg,
            target=target,
            skills_root=fold_sk_root,
            rulebook_text=fold_text,
            expect_version="v1",
            auth_mode=auth_mode,
            rationale=f"drafted from fold rulebook {tag}",
            log_dir=log_dir,
            stream=stream,
        )
        fold_cost += cost
        planned_fold = await _bench(
            cfg=cfg,
            target=target,
            skill=fold_skill,
            tasks=held_out_tasks,
            runs_root=fold_runs_root,
            splits=["test"],
            auth_mode=auth_mode,
            task_ids=None,
            max_concurrency=concurrency,
            on_start=on_bench_start,
            on_done=on_bench_done,
        )
        result = FoldResult(
            fold=fold,
            rulebooks_root=fold_rb_root,
            skills_root=fold_sk_root,
            runs_root=fold_runs_root,
            baseline_held_out=score(runs_root, _subset(planned_parent, fold.held_out, "test")),
            improved_held_out=score(fold_runs_root, planned_fold),
            cost_usd=fold_cost,
        )
        total_cost += fold_cost
        fold_results.append(result)
        if on_fold is not None:
            on_fold(result)

    # The refit: improve on every working task's evidence — the version carried forward.
    carried = await _ensure_rulebook(
        cfg=cfg,
        target=target,
        rulebooks_root=rulebooks_root,
        runs_root=runs_root,
        benched_skill=skill_parent,
        tasks=tasks,
        baseline_version=parent_version,
        baseline_text=parent_text,
        auth_mode=auth_mode,
        feedback=feedback,
        log_dir=log_dir,
        stream=stream,
        deny_dirs=deny_dirs,
        log_name=f"loop-rulebook-{carried_version}",
        expect_version=carried_version,
    )
    total_cost += carried.cost_usd
    carried_text = rb.load_rulebook(rulebooks_root, carried.version).text
    skill_carried, cost = await _ensure_skill(
        cfg=cfg,
        target=target,
        skills_root=skills_root,
        rulebook_text=carried_text,
        expect_version=f"v{version_number(carried.version)}",
        auth_mode=auth_mode,
        rationale=f"drafted from rulebook {carried.version}",
        log_dir=log_dir,
        stream=stream,
    )
    total_cost += cost
    planned_carried = await _bench(
        cfg=cfg,
        target=target,
        skill=skill_carried,
        tasks=tasks,
        runs_root=runs_root,
        splits=["test"],
        auth_mode=auth_mode,
        task_ids=None,
        max_concurrency=concurrency,
        on_start=on_bench_start,
        on_done=on_bench_done,
    )
    carried_test = score(runs_root, planned_carried)
    noskill_score = score(runs_root, build_matrix(cfg, tasks, skill=None, splits=["test"]))

    diff = "".join(
        difflib.unified_diff(
            parent_text.splitlines(keepends=True),
            carried_text.splitlines(keepends=True),
            fromfile=f"rulebook {parent_version}",
            tofile=f"rulebook {carried.version}",
        )
    )
    return CVResult(
        baseline_version=parent_version,
        baseline_skill=skill_parent.version,
        folds=fold_results,
        carried=carried,
        carried_skill=skill_carried.version,
        baseline_test=baseline_test,
        carried_test=carried_test,
        noskill_score=noskill_score,
        rulebook_diff=diff,
        cost_usd=total_cost,
        lockbox=lockbox,
        selection=selection,
    )


# ── The multi-iteration loop (P7) ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class StopRule:
    """When the outer loop stops. Every rule is a hard cap; the loop halts on the first one hit.

    ``patience`` is the scientific rule: stop once ``patience`` consecutive iterations fail to raise
    the cross-validated held-out rate above the best seen by more than ``min_delta``. The others
    are budget caps — iterations and wall-clock — so an unattended run cannot run away. Wall-clock
    is checked *between* iterations (an iteration is not interrupted; it is resumable anyway).
    """

    max_iterations: int = 5
    patience: int = 2
    min_delta: float = 0.0
    max_wallclock_s: float | None = None


@dataclass(frozen=True)
class LoopRun:
    """The outcome of the outer loop: the iterations, the version it chose, and the lockbox verdict."""

    iterations: list[CVResult]
    #: The rulebook version with the best cross-validated held-out rate — the loop's pick.
    best_version: str
    best_cv_rate: float
    stopped_because: str
    #: The chosen version's skill on the lockbox tasks (test split), scored once, after every
    #: selection was made. ``None`` without a lockbox.
    lockbox_score: Score | None = None
    #: The seed version's skill on the same lockbox tasks — the floor the pick is compared to.
    lockbox_baseline: Score | None = None
    cost_usd: float = 0.0

    @property
    def lockbox_delta(self) -> float | None:
        """Lockbox pass-rate change from the seed version to the pick — the one honest number."""
        if self.lockbox_score is None or self.lockbox_baseline is None:
            return None
        return self.lockbox_score.rate - self.lockbox_baseline.rate


def _cv_rates(result: CVResult) -> tuple[float, float]:
    """(parent, carried) cross-validated held-out pass rates — absolute, so versions compare."""
    if not result.folds:
        return 0.0, 0.0
    parent = sum(f.baseline_held_out.rate for f in result.folds) / len(result.folds)
    carried = sum(f.improved_held_out.rate for f in result.folds) / len(result.folds)
    return parent, carried


async def _lockbox_eval(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    runs_root: Path,
    version: str,
    tasks: Sequence[Task],
    auth_mode: AuthMode,
    max_concurrency: int,
    on_bench_start: Callable[[PlannedRun], None] | None,
    on_bench_done: Callable[[RunOutcome], None] | None,
) -> Score:
    """Score one skill version on the lockbox tasks' test split, into its own run tree.

    Resumable by file presence, which is what "scored once" means operationally: a version is
    benched on the lockbox at most once, however many times the loop is re-run.
    """
    skill = load_skill(skills_root, version, expect_name=cfg.skill_name)
    planned = await _bench(
        cfg=cfg,
        target=target,
        skill=skill,
        tasks=tasks,
        runs_root=runs_root / LOCKBOX_RUNS_DIRNAME,
        splits=["test"],
        auth_mode=auth_mode,
        task_ids=None,
        max_concurrency=max_concurrency,
        on_start=on_bench_start,
        on_done=on_bench_done,
    )
    return score(runs_root / LOCKBOX_RUNS_DIRNAME, planned)


async def run_loop(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    runs_root: Path,
    tasks: Sequence[Task],
    k: int = 3,
    seed: int = 0,
    stop: StopRule = StopRule(),
    lockbox_dir: Path | None = None,
    allow_no_lockbox: bool = False,
    auth_mode: AuthMode = "session",
    max_concurrency: int | None = None,
    task_ids: Sequence[str] | None = None,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    headroom_only: bool = False,
    clock: Callable[[], float] = time.monotonic,
    on_select: Callable[[HeadroomSelection], None] | None = None,
    on_iteration: Callable[[int, CVResult], None] | None = None,
    on_fold: Callable[[FoldResult], None] | None = None,
    on_bench_start: Callable[[PlannedRun], None] | None = None,
    on_bench_done: Callable[[RunOutcome], None] | None = None,
) -> LoopRun:
    """Iterate :func:`run_cv_iteration` until a :class:`StopRule` fires, pick by CV, then open the lockbox.

    Iteration ``i`` improves rulebook ``v{i}`` into ``v{i+1}`` (pinned, so a resumed loop replays the
    same chain from disk without re-running agents). After each, the carried version's
    cross-validated held-out rate is compared with the best so far; the loop stops on ``patience``
    non-improving iterations, on ``max_iterations``, or when the wall-clock cap would be exceeded.

    Selection happens on CV scores only — which is exactly why they are optimistic — and then, once,
    the chosen version and the seed version are benched on the **lockbox** tasks, a set nothing in
    the loop ever read. The lockbox delta is the loop's one honest generalisation number; the CV
    numbers are its working estimates.
    """
    started = clock()
    concurrency = max_concurrency or cfg.max_concurrency
    history: list[CVResult] = []
    best_version = rb.seed_default(rulebooks_root)
    best_rate = -1.0
    without_improvement = 0
    reason = f"reached max_iterations={stop.max_iterations}"
    total_cost = 0.0

    for i in range(stop.max_iterations):
        if stop.max_wallclock_s is not None and clock() - started >= stop.max_wallclock_s:
            reason = f"wall-clock cap of {stop.max_wallclock_s:.0f}s reached before iteration {i + 1}"
            break
        parent = f"v{i + 1}"
        result = await run_cv_iteration(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            rulebooks_root=rulebooks_root,
            runs_root=runs_root,
            tasks=tasks,
            k=k,
            seed=seed,
            lockbox_dir=lockbox_dir,
            allow_no_lockbox=allow_no_lockbox,
            parent_version=parent,
            auth_mode=auth_mode,
            max_concurrency=concurrency,
            task_ids=task_ids,
            feedback=feedback,
            log_dir=log_dir,
            stream=stream,
            headroom_only=headroom_only and i == 0,
            on_select=on_select,
            on_fold=on_fold,
            on_bench_start=on_bench_start,
            on_bench_done=on_bench_done,
        )
        history.append(result)
        total_cost += result.cost_usd
        if result.selection is not None:
            # The headroom selection is made once, on the first iteration; later iterations must
            # score the same tasks or the CV numbers are not comparable across versions.
            tasks = result.selection.selected
            task_ids = None
        parent_rate, carried_rate = _cv_rates(result)
        if i == 0:
            best_rate = parent_rate  # the seed's own CV rate is the bar the first improvement must clear
        if on_iteration is not None:
            on_iteration(i + 1, result)
        if carried_rate > best_rate + stop.min_delta:
            best_version, best_rate = result.carried.version, carried_rate
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= stop.patience:
                reason = f"no CV improvement over {best_version} for {stop.patience} iteration(s)"
                break

    lockbox_score = lockbox_baseline = None
    box = history[-1].lockbox if history else None
    if box is not None:
        lock_tasks = load_lockbox_tasks(box)
        common = dict(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            runs_root=runs_root,
            tasks=lock_tasks,
            auth_mode=auth_mode,
            max_concurrency=concurrency,
            on_bench_start=on_bench_start,
            on_bench_done=on_bench_done,
        )
        lockbox_baseline = await _lockbox_eval(version="v1", **common)
        lockbox_score = (
            lockbox_baseline if best_version == "v1" else await _lockbox_eval(version=best_version, **common)
        )

    return LoopRun(
        iterations=history,
        best_version=best_version,
        best_cv_rate=max(best_rate, 0.0),
        stopped_because=reason,
        lockbox_score=lockbox_score,
        lockbox_baseline=lockbox_baseline,
        cost_usd=total_cost,
    )
