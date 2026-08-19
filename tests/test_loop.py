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
from acumen.loop import LoopResult, RulebookResult, run_iteration, score
from acumen.paths import RESULT_FILE, RunKey, run_dir
from acumen.prompts import DRAFT_PROMPT, draft_prompt
from acumen.rulebooks import RulebookError, seed_default, validate_rulebook
from acumen.skills import load_skill, next_version
from acumen.tasks import Task, TaskSplit

MODEL = "claude-haiku-4-5-20251001"


# ── Rulebook artifact ───────────────────────────────────────────────────────────────────


def test_seed_default_is_idempotent_and_reproduces_the_draft_prompt(tmp_path: Path) -> None:
    assert seed_default(tmp_path) == "v1"
    assert seed_default(tmp_path) == "v1"  # idempotent — does not create v2
    assert rb.load_rulebook(tmp_path, "v1") == DRAFT_PROMPT
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
    assert draft_prompt(**kwargs) == draft_prompt(**kwargs, template=rb.load_rulebook(tmp_path, "v1"))


def test_validate_rulebook_rejects_broken_templates_and_accepts_the_default() -> None:
    with pytest.raises(RulebookError, match="required placeholder"):
        validate_rulebook("no placeholders at all")
    with pytest.raises(RulebookError, match="required placeholder"):
        validate_rulebook("only {skill_name} here")  # missing {out}
    with pytest.raises(RulebookError, match="cannot fill"):
        validate_rulebook("{out} {skill_name} but also {bogus}")  # stray field
    validate_rulebook(DRAFT_PROMPT)  # the real template passes


def test_write_rulebook_is_immutable(tmp_path: Path) -> None:
    rb.write_rulebook(tmp_path, "v1", DRAFT_PROMPT)
    with pytest.raises(RulebookError, match="immutable"):
        rb.write_rulebook(tmp_path, "v1", DRAFT_PROMPT)


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
        text = rb.load_rulebook(rulebooks_root, parent_version) + "\n<!-- tweak -->\n"
        rb.write_rulebook(rulebooks_root, "v2", text)
        (rb.rulebook_dir(rulebooks_root, "v2") / "rationale.md").write_text("made the description guidance stronger\n")
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


def _run(tmp_path: Path, cfg: Config, calls: dict) -> LoopResult:
    return asyncio.run(
        run_iteration(
            cfg=cfg,
            target=_target(tmp_path),
            skills_root=tmp_path / "skills",
            rulebooks_root=tmp_path / "rulebooks",
            runs_root=tmp_path / "runs",
            tasks=[_task("only")],
            auth_mode="session",
        )
    )


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
