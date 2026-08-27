"""Tests for transient-failure handling: a platform limit must never become a recorded run.

When the subscription session limit (or a rate limit) hits mid-pass, the first live CV loop wrote
54 "failed" result.json files in minutes — runs that said nothing about the tasks, that resume would
then skip forever, and that the improver would read as evidence. These tests pin the fix: such an
outcome is flagged transient and not written, the matrix stops launching into the wall, and the pass
ends by raising so no caller scores a partial matrix.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from acumen import bench as bench_mod
from acumen.bench import build_matrix, run_matrix
from acumen.config import Config
from acumen.env import Target
from acumen.paths import RESULT_FILE, run_dir
from acumen.runner import RunOutcome, TransientLimitError, is_transient
from acumen.tasks import Task, TaskSplit

MODEL = "claude-sonnet-5"


def test_is_transient_recognises_platform_limits() -> None:
    assert is_transient(
        "ResultError: Claude Code returned an error result: You've hit your session limit · resets 8:40pm"
    )
    assert is_transient("API rate limit exceeded")
    assert is_transient("overloaded_error: Overloaded")
    assert is_transient("HTTP 529 upstream")
    assert not is_transient("ValueError: no answer file")
    assert not is_transient("the agent wrote the wrong answer")


def _tasks(n: int) -> list[Task]:
    return [Task(id=f"t{i}", train=TaskSplit("p", "A"), test=TaskSplit("p", "B")) for i in range(n)]


def test_run_matrix_pauses_after_a_transient_failure_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(repo="/pkg", skill_name="pkg", models=[MODEL], n_replicates=1, max_concurrency=1)
    target = Target(
        source="x", ref="main", src_dir=tmp_path, venv_dir=tmp_path / "venv", commit="c", pkg_name="p", pkg_version="1"
    )
    planned = build_matrix(cfg, _tasks(4), skill=None, splits=["test"])
    ran: list[str] = []

    async def fake_run_once(*, key, run_dir, **_):
        # Launch order across tasks is arbitrary (as_completed), so key off the call count: the
        # first run is genuine, the second hits the limit, and nothing is written for it.
        ran.append(key.task_id)
        if len(ran) == 2:
            return RunOutcome(
                key=key, success=False, reason="error", payload={"transient": True, "error": "session limit"}
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / RESULT_FILE).write_text('{"success": true}')
        return RunOutcome(key=key, success=True, reason="ok", payload={})

    monkeypatch.setattr(bench_mod, "run_once", fake_run_once)

    with pytest.raises(TransientLimitError, match="3 of 4 runs were not recorded"):
        asyncio.run(
            run_matrix(planned, target=target, runs_root=tmp_path / "runs", max_concurrency=1, auth_mode="session")
        )

    # Concurrency 1: one genuine run, one that hit the wall, and the other two were never launched.
    assert len(ran) == 2
    recorded = {p.parents[1].name for p in (tmp_path / "runs").rglob(RESULT_FILE)}  # .../<task>/rep_1/result.json
    assert recorded == {ran[0]}  # only the genuine run left a result.json
    assert not (
        run_dir(tmp_path / "runs", next(p for p in planned if p.key.task_id == ran[1]).key) / RESULT_FILE
    ).exists()
