"""Tests for the crude autoresearch-loop prototype — rulebook artifact, scoring, orchestration.

Like the rest of the suite these never spawn an agent. The loop's three agent boundaries
(``draft_skill``, ``run_matrix``, ``improve_rulebook``) are monkeypatched, so ``run_iteration``'s
own logic — seeding v1, the rulebook<->skill lockstep, scoring from disk, the moved calculation,
and resume — is exercised while the SDK is never touched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from acumen import loop as loop_mod
from acumen import rulebooks as rb
from acumen.bench import PlannedRun
from acumen.config import Config
from acumen.env import Target
from acumen.folds import write_lockbox
from acumen.loop import (
    DraftScores,
    LoopError,
    LoopResult,
    RulebookResult,
    Score,
    StopRule,
    run_cv_iteration,
    run_iteration,
    run_loop,
    score,
)
from acumen.paths import RESULT_FILE, RunKey, run_dir
from acumen.prompts import DRAFT_PROMPT, draft_prompt
from acumen.rulebooks import RulebookError, seed_default, validate_rulebook
from acumen.skills import load_skill, next_version
from acumen.taskgen import dump_tasks
from acumen.tasks import Task, TaskSplit

MODEL = "claude-haiku-4-5-20251001"


# ── Rulebook artifact ───────────────────────────────────────────────────────────────────


def test_seed_default_is_idempotent_and_reproduces_the_draft_prompt(tmp_path: Path) -> None:
    assert seed_default(tmp_path) == "v1"
    assert seed_default(tmp_path) == "v1"  # idempotent — does not create v2
    assert rb.load_rulebook(tmp_path, "v1").text == DRAFT_PROMPT
    assert rb.next_version(tmp_path) == "v2"


def test_draft_prompt_with_v1_rulebook_is_byte_identical_to_the_default(tmp_path: Path) -> None:
    seed_default(tmp_path)
    kwargs = {
        "package": "squidpy",
        "version": "1.0",
        "src": Path("/s"),
        "python": Path("/p"),
        "out": Path("/o"),
        "skill_name": "squidpy",
    }
    assert draft_prompt(**kwargs) == draft_prompt(**kwargs, template=rb.load_rulebook(tmp_path, "v1").text)


def test_validate_rulebook_rejects_broken_templates_and_accepts_the_default() -> None:
    with pytest.raises(RulebookError, match="required placeholder"):
        validate_rulebook("no placeholders at all")
    with pytest.raises(RulebookError, match="required placeholder"):
        validate_rulebook("only {skill_name} here")  # missing {out}
    with pytest.raises(RulebookError, match="cannot fill"):
        validate_rulebook("{out} {skill_name} but also {bogus}")  # stray field
    validate_rulebook(DRAFT_PROMPT)  # the real template passes


def test_write_rulebook_is_immutable(tmp_path: Path) -> None:
    rb.write_rulebook(tmp_path, "v1", DRAFT_PROMPT, parent=None, rationale="seed")
    with pytest.raises(RulebookError, match="immutable"):
        rb.write_rulebook(tmp_path, "v1", DRAFT_PROMPT, parent=None, rationale="seed")


# ── Scoring ─────────────────────────────────────────────────────────────────────────────


def _planned(arm: str, split: str, task_id: str, rep: int = 1) -> PlannedRun:
    key = RunKey(arm=arm, split=split, model=MODEL, task_id=task_id, rep=rep)
    return PlannedRun(key=key, task=_task(task_id), model=MODEL, max_turns=1, max_usd=1.0)


def _write_result(runs_root: Path, planned: PlannedRun, *, success: bool, loaded: bool = True) -> None:
    directory = run_dir(runs_root, planned.key)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULT_FILE).write_text(
        json.dumps({"success": success, "skill_loaded": loaded, "model": MODEL, "answer": "x", "reason": "ok"})
    )


def test_score_counts_passes_and_loads_over_complete_runs(tmp_path: Path) -> None:
    passing = _planned("skill_v1", "test", "a")  # passed and loaded
    failing = _planned("skill_v1", "test", "b")  # failed but loaded
    noload = _planned("skill_v1", "test", "c")  # failed and never loaded
    missing = _planned("skill_v1", "test", "d")  # no result.json written
    _write_result(tmp_path, passing, success=True, loaded=True)
    _write_result(tmp_path, failing, success=False, loaded=True)
    _write_result(tmp_path, noload, success=False, loaded=False)

    result = score(tmp_path, [passing, failing, noload, missing])

    # pass and load are independent metrics: 1/3 passed, 2/3 loaded.
    assert (result.passed, result.total, result.loaded) == (1, 3, 2)
    assert result.rate == pytest.approx(1 / 3)
    assert result.load_rate == pytest.approx(2 / 3)


# ── Orchestration (agents mocked) ───────────────────────────────────────────────────────


def _task(task_id: str = "t") -> Task:
    return Task(id=task_id, train=TaskSplit("train goal", "TRAIN"), test=TaskSplit("test goal", "TEST"))


def _write_skill(skills_root: Path, version: str, name: str) -> None:
    directory = skills_root / version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use {name} for its analyses.\n---\n\n# {name}\n\nCall it.\n"
    )


def _target(tmp_path: Path) -> Target:
    return Target(
        source="/pkg",
        ref="main",
        src_dir=tmp_path / "src",
        venv_dir=tmp_path / "venv",
        commit="abc123",
        pkg_name="pkg",
        pkg_version="1.0",
    )


def _install_fakes(monkeypatch: pytest.MonkeyPatch, cfg: Config, success_by_arm_split: dict, calls: dict) -> None:
    """Monkeypatch the loop's three agent boundaries with filesystem-writing fakes."""

    async def fake_draft(*, cfg, target, skills_root, rulebook, **_):
        calls["draft"] += 1
        version = next_version(skills_root)
        _write_skill(skills_root, version, cfg.skill_name)
        skill = load_skill(skills_root, version, expect_name=cfg.skill_name)
        return SimpleNamespace(skill=skill, cost_usd=0.1)

    async def fake_run_matrix(planned, *, runs_root, **_):
        calls["bench"] += 1
        for item in planned:
            _write_result(runs_root, item, success=success_by_arm_split[(item.key.arm, item.key.split)])
        return []

    async def fake_improve(*, rulebooks_root, parent_version, **_):
        calls["improve"] += 1
        text = rb.load_rulebook(rulebooks_root, parent_version).text + "\n<!-- tweak -->\n"
        rb.write_rulebook(
            rulebooks_root, "v2", text, parent=parent_version, rationale="made the description guidance stronger"
        )
        return RulebookResult(
            version="v2",
            parent=parent_version,
            path=rb.rulebook_dir(rulebooks_root, "v2") / rb.RULEBOOK_FILE,
            rationale="made the description guidance stronger",
            changed=True,
            cost_usd=0.2,
            turns=1,
            n_train_runs=1,
            n_train_failures=1,
        )

    monkeypatch.setattr(loop_mod, "draft_skill", fake_draft)
    monkeypatch.setattr(loop_mod, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(loop_mod, "improve_rulebook", fake_improve)


def _run(
    tmp_path: Path,
    cfg: Config,
    calls: dict,
    *,
    tasks: list[Task] | None = None,
    headroom_only: bool = False,
    on_select=None,
) -> LoopResult:
    return asyncio.run(
        run_iteration(
            cfg=cfg,
            target=_target(tmp_path),
            skills_root=tmp_path / "skills",
            rulebooks_root=tmp_path / "rulebooks",
            runs_root=tmp_path / "runs",
            tasks=tasks or [_task("only")],
            auth_mode="session",
            headroom_only=headroom_only,
            on_select=on_select,
        )
    )


def _write_baseline(runs_root: Path, task_id: str, *, success: bool, split: str = "test") -> None:
    """A prior `acumen bench --no-skill` result for one task, as headroom selection reads it."""
    key = RunKey(arm="noskill", split=split, model=MODEL, task_id=task_id, rep=1)
    directory = run_dir(runs_root, key)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULT_FILE).write_text(json.dumps({"success": success, "skill_loaded": False, "model": MODEL}))


def test_run_iteration_moves_the_held_out_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    # v1 fails the held-out (test) variant but passes train; v2 passes held-out — a clear +1 move.
    success = {("skill_v1", "train"): True, ("skill_v1", "test"): False, ("skill_v2", "test"): True}
    _install_fakes(monkeypatch, cfg, success, calls)

    result = _run(tmp_path, cfg, calls)

    assert (result.baseline_version, result.improved_version) == ("v1", "v2")
    assert (result.baseline_skill, result.improved_skill) == ("v1", "v2")
    assert (result.baseline_score.passed, result.baseline_score.total) == (0, 1)
    assert (result.improved_score.passed, result.improved_score.total) == (1, 1)
    assert result.baseline_train_score.passed == 1
    assert result.moved == 1
    # Load rate plumbs through: the fake marks every run loaded, so both skill arms load 1/1.
    assert result.baseline_score.loaded == 1 and result.improved_score.loaded == 1
    assert result.load_moved == 0
    # No noskill arm was benched into runs_root by the fakes, so the floor reads as absent.
    assert result.noskill_score.total == 0
    assert result.rulebook.changed
    assert result.rulebook_diff.strip()  # a real unified diff was produced
    assert result.cost_usd == pytest.approx(0.4)  # 0.1 draft + 0.2 improve + 0.1 draft
    assert calls == {"draft": 2, "bench": 2, "improve": 1}


def test_run_iteration_headroom_only_scores_just_the_tasks_the_baseline_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --headroom the loop narrows to baseline-failed tasks BEFORE any agent runs.

    'easy' is baseline-solved (no room to show movement), 'never' was never screened (cannot be
    placed, so excluded rather than guessed at); only 'hard' is benched.
    """
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    success = {("skill_v1", "train"): True, ("skill_v1", "test"): False, ("skill_v2", "test"): True}
    _install_fakes(monkeypatch, cfg, success, calls)
    runs = tmp_path / "runs"
    _write_baseline(runs, "easy", success=True)
    _write_baseline(runs, "hard", success=False)
    seen: list[tuple[list[str], int]] = []

    def on_select(selection) -> None:
        # Recorded with the draft count at the time: the decision precedes every agent.
        seen.append(([t.id for t in selection.selected], calls["draft"]))

    result = _run(
        tmp_path,
        cfg,
        calls,
        tasks=[_task("easy"), _task("hard"), _task("never")],
        headroom_only=True,
        on_select=on_select,
    )

    assert seen == [(["hard"], 0)]
    assert result.selection is not None
    assert [t.id for t in result.selection.selected] == ["hard"]
    assert result.selection.solved == ["easy"] and result.selection.unscreened == ["never"]
    # Only the selected task was benched in either skill arm.
    benched = {p.name for arm in ("skill_v1", "skill_v2") for p in (runs / arm / "test").rglob("rep_1")}
    assert benched == {"rep_1"}
    assert {p.parent.name for arm in ("skill_v1", "skill_v2") for p in (runs / arm / "test").rglob("rep_1")} == {"hard"}
    assert (result.baseline_score.total, result.improved_score.total) == (1, 1)
    # The floor is read for the same selected task only.
    assert (result.noskill_score.passed, result.noskill_score.total) == (0, 1)


def test_run_iteration_headroom_only_refuses_when_nothing_can_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    _install_fakes(monkeypatch, cfg, {}, calls)
    _write_baseline(tmp_path / "runs", "easy", success=True)

    with pytest.raises(LoopError, match="no task has headroom"):
        _run(tmp_path, cfg, calls, tasks=[_task("easy"), _task("never")], headroom_only=True)
    assert calls == {"draft": 0, "bench": 0, "improve": 0}  # refused before spending anything


# ── Cross-validated iteration (P5) ──────────────────────────────────────────────────────


def _install_cv_fakes(monkeypatch: pytest.MonkeyPatch, cfg: Config, calls: dict, improves: list[dict]) -> None:
    """Fakes for the CV loop: every skill fails held-out tests until improved, then passes.

    ``improves`` records each improve invocation's boundary kwargs, which is what the tests assert
    on: the CV guarantee is *what evidence the agent was given and what it was denied*.
    """

    async def fake_draft(*, cfg, target, skills_root, rulebook, **_):
        calls["draft"] += 1
        version = next_version(skills_root)
        _write_skill(skills_root, version, cfg.skill_name)
        return SimpleNamespace(skill=load_skill(skills_root, version, expect_name=cfg.skill_name), cost_usd=0.1)

    async def fake_run_matrix(planned, *, runs_root, **_):
        calls["bench"] += 1
        for item in planned:
            # Main tree: parent skill passes train, fails test. CV trees: fold skills pass.
            in_cv = loop_mod.CV_DIRNAME in runs_root.parts
            _write_result(runs_root, item, success=in_cv or item.key.split == "train")
        return []

    async def fake_improve(*, rulebooks_root, parent_version, tasks, held_out_ids=(), deny_dirs=(), **_):
        calls["improve"] += 1
        improves.append(
            {
                "root": rulebooks_root,
                "tasks": sorted(t.id for t in tasks),
                "held_out": tuple(held_out_ids),
                "deny": [Path(d) for d in deny_dirs],
            }
        )
        text = rb.load_rulebook(rulebooks_root, parent_version).text + f"\n<!-- {rulebooks_root.name} -->\n"
        new = rb.next_version(rulebooks_root)
        rb.write_rulebook(rulebooks_root, new, text, parent=parent_version, rationale="tweak")
        return RulebookResult(
            version=new,
            parent=parent_version,
            path=rb.rulebook_dir(rulebooks_root, new) / rb.RULEBOOK_FILE,
            rationale="tweak",
            changed=True,
            cost_usd=0.2,
            turns=1,
            n_train_runs=len(tasks),
            n_train_failures=0,
        )

    monkeypatch.setattr(loop_mod, "draft_skill", fake_draft)
    monkeypatch.setattr(loop_mod, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(loop_mod, "improve_rulebook", fake_improve)


def _cv(tmp_path: Path, cfg: Config, tasks: list[Task], **kw):
    return asyncio.run(
        run_cv_iteration(
            cfg=cfg,
            target=_target(tmp_path),
            skills_root=tmp_path / "skills",
            rulebooks_root=tmp_path / "rulebooks",
            runs_root=tmp_path / "runs",
            tasks=tasks,
            auth_mode="session",
            **kw,
        )
    )


def test_cv_iteration_holds_out_structurally_and_estimates_over_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    improves: list[dict] = []
    _install_cv_fakes(monkeypatch, cfg, calls, improves)
    tasks = [_task(t) for t in ("a", "b", "c", "d")]
    lockbox = write_lockbox(tmp_path / "lockbox", [_task("z")], seed=0, fraction=0.2, dump=dump_tasks)

    result = _cv(tmp_path, cfg, tasks, k=2, seed=0, lockbox_dir=lockbox.directory)

    # Two fold estimates plus the refit: three improves, each with the right boundary.
    assert calls["improve"] == 3 and len(result.folds) == 2
    fold_calls = [c for c in improves if loop_mod.CV_DIRNAME in c["root"].parts]
    refit = [c for c in improves if loop_mod.CV_DIRNAME not in c["root"].parts]
    assert len(fold_calls) == 2 and len(refit) == 1
    for fold, call in zip(result.folds, fold_calls, strict=True):
        assert call["tasks"] == sorted(fold.fold.optimize)  # evidence: optimize tasks only
        assert set(call["held_out"]) == set(fold.fold.held_out)  # guard: held-out tasks denied
        assert set(fold.fold.optimize) | set(fold.fold.held_out) == {"a", "b", "c", "d"}
        assert any(d.name == loop_mod.CV_DIRNAME for d in call["deny"])  # other folds' trees denied
        assert lockbox.directory in call["deny"]  # the lockbox denied
    assert refit[0]["tasks"] == ["a", "b", "c", "d"] and refit[0]["held_out"] == ()

    # Each fold is scored on exactly its held-out tasks; baseline fails them, the fold skill passes.
    for f in result.folds:
        assert f.improved_held_out.total == len(f.fold.held_out) == f.baseline_held_out.total
        assert f.baseline_held_out.passed == 0 and f.improved_held_out.passed == len(f.fold.held_out)
        assert f.delta_rate == 1.0
    assert result.cv_mean_delta == 1.0 and result.cv_spread == 0.0
    # Fold artifacts live in their own roots; the linear chains hold only v1 -> v2 (the refit).
    assert rb.available_versions(tmp_path / "rulebooks") == ["v1", "v2"]
    assert result.carried.version == "v2" and result.carried_skill == "v2"
    assert (tmp_path / "rulebooks" / loop_mod.CV_DIRNAME / "v2" / "fold-1" / "v2").is_dir()
    assert result.lockbox == lockbox


def test_cv_iteration_requires_a_lockbox_and_refuses_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    _install_cv_fakes(monkeypatch, cfg, calls, [])
    tasks = [_task(t) for t in ("a", "b", "c", "d")]

    with pytest.raises(LoopError, match="no lockbox"):
        _cv(tmp_path, cfg, tasks, k=2)
    lockbox = write_lockbox(tmp_path / "lockbox", [_task("a")], seed=0, fraction=0.2, dump=dump_tasks)
    with pytest.raises(LoopError, match="in the lockbox"):
        _cv(tmp_path, cfg, tasks, k=2, lockbox_dir=lockbox.directory)
    assert calls == {"draft": 0, "bench": 0, "improve": 0}  # refused before any agent ran
    # Explicitly waiving the lockbox is allowed, and recorded as absent.
    result = _cv(tmp_path, cfg, tasks, k=2, allow_no_lockbox=True)
    assert result.lockbox is None and len(result.folds) == 2


def test_run_loop_stops_on_patience_picks_by_cv_and_opens_the_lockbox_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Iteration 1 lifts the CV rate 0 -> 1; nothing can beat that, so patience=1 stops after iteration 2.

    The pick is v2 (best CV), and the lockbox is benched exactly once for v1 and once for v2 —
    on the lockbox tasks only, in its own run tree — after all selection is done.
    """
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    improves: list[dict] = []
    _install_cv_fakes(monkeypatch, cfg, calls, improves)
    tasks = [_task(t) for t in ("a", "b", "c", "d")]
    lockbox = write_lockbox(tmp_path / "lockbox", [_task("z1"), _task("z2")], seed=0, fraction=0.3, dump=dump_tasks)
    seen: list[tuple[int, str]] = []

    run = asyncio.run(
        run_loop(
            cfg=cfg,
            target=_target(tmp_path),
            skills_root=tmp_path / "skills",
            rulebooks_root=tmp_path / "rulebooks",
            runs_root=tmp_path / "runs",
            tasks=tasks,
            k=2,
            stop=StopRule(max_iterations=5, patience=1),
            lockbox_dir=lockbox.directory,
            auth_mode="session",
            on_iteration=lambda i, r: seen.append((i, r.carried.version)),
        )
    )

    assert seen == [(1, "v2"), (2, "v3")]
    assert run.best_version == "v2" and run.best_cv_rate == 1.0
    assert "no CV improvement" in run.stopped_because
    assert rb.available_versions(tmp_path / "rulebooks") == ["v1", "v2", "v3"]
    # Iteration 2's parent was pinned to v2 (not "latest"), so its fold trees sit under v3/.
    assert (tmp_path / "rulebooks" / loop_mod.CV_DIRNAME / "v3" / "fold-1").is_dir()
    # Lockbox: only z1/z2, only the test split, only v1 and v2, in runs/lockbox/.
    lock_runs = tmp_path / "runs" / loop_mod.LOCKBOX_RUNS_DIRNAME
    benched = {(p.parts[-5], p.parts[-4], p.parts[-2]) for p in lock_runs.rglob("rep_1")}
    assert benched == {
        ("skill_v1", "test", "z1"),
        ("skill_v1", "test", "z2"),
        ("skill_v2", "test", "z1"),
        ("skill_v2", "test", "z2"),
    }
    assert run.lockbox_baseline is not None and run.lockbox_baseline.total == 2
    assert run.lockbox_score is not None and run.lockbox_score.total == 2
    assert run.lockbox_delta == 0.0  # the fakes fail every non-CV test run, so the honest number is flat

    # A resumed run replays the same chain from disk: same pick, same stop, no agents spawned.
    calls.update(draft=0, bench=0, improve=0)
    again = asyncio.run(
        run_loop(
            cfg=cfg,
            target=_target(tmp_path),
            skills_root=tmp_path / "skills",
            rulebooks_root=tmp_path / "rulebooks",
            runs_root=tmp_path / "runs",
            tasks=tasks,
            k=2,
            stop=StopRule(max_iterations=5, patience=1),
            lockbox_dir=lockbox.directory,
            auth_mode="session",
        )
    )
    assert (again.best_version, again.stopped_because) == (run.best_version, run.stopped_because)
    assert calls["draft"] == 0 and calls["improve"] == 0


def test_run_loop_wallclock_cap_stops_between_iterations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    _install_cv_fakes(monkeypatch, cfg, calls, [])
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])  # started, check#1 (ok), check#2 (over)

    run = asyncio.run(
        run_loop(
            cfg=cfg,
            target=_target(tmp_path),
            skills_root=tmp_path / "skills",
            rulebooks_root=tmp_path / "rulebooks",
            runs_root=tmp_path / "runs",
            tasks=[_task(t) for t in ("a", "b", "c", "d")],
            k=2,
            stop=StopRule(max_iterations=5, patience=5, max_wallclock_s=50),
            allow_no_lockbox=True,
            auth_mode="session",
            clock=lambda: next(ticks),
        )
    )

    assert len(run.iterations) == 1 and "wall-clock" in run.stopped_because
    assert run.lockbox_score is None and run.lockbox_delta is None


def test_run_iteration_resumes_without_respawning_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    success = {("skill_v1", "train"): True, ("skill_v1", "test"): False, ("skill_v2", "test"): True}
    _install_fakes(monkeypatch, cfg, success, calls)

    first = _run(tmp_path, cfg, calls)
    calls.update(draft=0, bench=0, improve=0)
    second = _run(tmp_path, cfg, calls)

    # Everything was already on disk: no skill drafted, no bench run, no rulebook improved.
    assert calls == {"draft": 0, "bench": 0, "improve": 0}
    # ...yet the scores and versions reproduce exactly from disk.
    assert second.moved == first.moved == 1
    assert (second.baseline_version, second.improved_version) == ("v1", "v2")
    assert second.rulebook.changed  # reconstructed from the on-disk rulebook, not the baseline


def test_draft_refused_by_the_platform_is_a_pause_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from acumen.draft import DraftError
    from acumen.runner import TransientLimitError

    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    _install_fakes(monkeypatch, cfg, {}, calls)

    async def refused(**_):
        raise DraftError("the drafting agent failed: ResultError: You've hit your session limit · resets 8:50am")

    monkeypatch.setattr(loop_mod, "draft_skill", refused)
    with pytest.raises(TransientLimitError, match="rerun to resume"):
        _run(tmp_path, cfg, calls)
    assert not (tmp_path / "skills" / "v1").exists()  # nothing written; resume starts at the draft


# ── N-draft scoring (draft variance as a reported quantity) ─────────────────────────────


def test_draft_scores_mean_and_spread() -> None:
    ds = DraftScores(
        version="v2",
        scores=[Score(passed=27, total=36), Score(passed=22, total=36), Score(passed=29, total=36)],
        sizes=[100, 200, 300],
    )
    assert ds.mean_rate == pytest.approx((27 + 22 + 29) / (3 * 36))
    assert ds.spread == pytest.approx((29 - 22) / 36)
    empty = DraftScores(version="v1", scores=[], sizes=[])
    assert empty.mean_rate == 0.0 and empty.spread == 0.0 and empty.mean_load_rate == 0.0


def test_run_loop_rejects_a_draft_count_below_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1)
    _install_cv_fakes(monkeypatch, cfg, {"draft": 0, "bench": 0, "improve": 0}, [])
    with pytest.raises(LoopError, match="n_drafts"):
        asyncio.run(
            run_loop(
                cfg=cfg,
                target=_target(tmp_path),
                skills_root=tmp_path / "skills",
                rulebooks_root=tmp_path / "rulebooks",
                runs_root=tmp_path / "runs",
                tasks=[_task(t) for t in ("a", "b", "c", "d")],
                k=2,
                n_drafts=0,
                allow_no_lockbox=True,
                auth_mode="session",
            )
        )


def test_run_loop_n_drafts_scores_the_lockbox_over_variants_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --drafts 3, v1 and the pick each get three independent drafts of the SAME rulebook text,
    every draft is benched once on the lockbox in its own run tree, the primary draft keeps its
    single-draft paths/fields, and a re-run replays everything from disk without spawning agents."""
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=2)
    calls = {"draft": 0, "bench": 0, "improve": 0}
    improves: list[dict] = []
    _install_cv_fakes(monkeypatch, cfg, calls, improves)
    tasks = [_task(t) for t in ("a", "b", "c", "d")]
    lockbox = write_lockbox(tmp_path / "lockbox", [_task("z1"), _task("z2")], seed=0, fraction=0.3, dump=dump_tasks)
    kw = {
        "cfg": cfg,
        "target": _target(tmp_path),
        "skills_root": tmp_path / "skills",
        "rulebooks_root": tmp_path / "rulebooks",
        "runs_root": tmp_path / "runs",
        "tasks": tasks,
        "k": 2,
        "n_drafts": 3,
        "stop": StopRule(max_iterations=5, patience=1),
        "lockbox_dir": lockbox.directory,
        "auth_mode": "session",
    }

    run = asyncio.run(run_loop(**kw))

    assert run.best_version == "v2"
    for version in ("v1", "v2"):
        for i in (2, 3):
            # Layout: each extra draft is its own skills-root holding a single v1 ...
            skill_root = loop_mod.draft_variant_root(tmp_path / "skills", version, i)
            assert (skill_root / "v1" / "SKILL.md").is_file()
            # ... benched on exactly the lockbox tasks, test split, in its own run tree.
            runs = loop_mod.draft_variant_root(tmp_path / "runs", version, i) / loop_mod.LOCKBOX_RUNS_DIRNAME
            benched = {(p.parts[-5], p.parts[-4], p.parts[-2]) for p in runs.rglob("rep_1")}
            assert benched == {("skill_v1", "test", "z1"), ("skill_v1", "test", "z2")}
    # The primary drafts' lockbox runs stay exactly where single-draft mode put them.
    lock_runs = tmp_path / "runs" / loop_mod.LOCKBOX_RUNS_DIRNAME
    primary = {(p.parts[-5], p.parts[-2]) for p in lock_runs.rglob("rep_1")}
    assert primary == {("skill_v1", "z1"), ("skill_v1", "z2"), ("skill_v2", "z1"), ("skill_v2", "z2")}

    assert run.lockbox_drafts is not None and run.lockbox_baseline_drafts is not None
    assert len(run.lockbox_drafts.scores) == len(run.lockbox_baseline_drafts.scores) == 3
    assert all(s.total == 2 for s in run.lockbox_drafts.scores + run.lockbox_baseline_drafts.scores)
    assert all(size > 0 for size in run.lockbox_drafts.sizes + run.lockbox_baseline_drafts.sizes)
    # The compat fields are the primary draft's scores; the mean delta is defined and mean-based.
    assert run.lockbox_score == run.lockbox_drafts.scores[0]
    assert run.lockbox_baseline == run.lockbox_baseline_drafts.scores[0]
    assert run.lockbox_mean_delta == run.lockbox_drafts.mean_rate - run.lockbox_baseline_drafts.mean_rate
    # Every improve agent was denied the drafts run tree, alongside the CV trees and the lockbox.
    assert improves and all(any(d.name == loop_mod.DRAFTS_DIRNAME for d in c["deny"]) for c in improves)

    # Resume: everything is on disk, so no draft, bench, or improve agent runs again.
    calls.update(draft=0, bench=0, improve=0)
    again = asyncio.run(run_loop(**kw))
    assert calls == {"draft": 0, "bench": 0, "improve": 0}
    assert again.lockbox_mean_delta == run.lockbox_mean_delta
    assert again.lockbox_drafts == run.lockbox_drafts
