"""Tests for baseline difficulty screening — pure aggregation over the run tree."""

from __future__ import annotations

import json
from pathlib import Path

from acumen.difficulty import FLAKY, HARD, SOLVED, Difficulty, screen
from acumen.paths import RESULT_FILE, RunKey, run_dir
from acumen.tasks import Task, TaskSplit

MODEL = "claude-haiku-4-5-20251001"


def _task(task_id: str) -> Task:
    return Task(id=task_id, train=TaskSplit("p", "A"), test=TaskSplit("p", "B"))


def _write(runs: Path, task_id: str, split: str, rep: int, success: bool) -> None:
    key = RunKey(arm="noskill", split=split, model=MODEL, task_id=task_id, rep=rep)
    d = run_dir(runs, key)
    d.mkdir(parents=True, exist_ok=True)
    (d / RESULT_FILE).write_text(json.dumps({"success": success, "model": MODEL}))


def test_difficulty_strata() -> None:
    assert Difficulty("t", "test", 0, 3).stratum == HARD
    assert Difficulty("t", "test", 3, 3).stratum == SOLVED
    assert Difficulty("t", "test", 1, 3).stratum == FLAKY
    assert Difficulty("t", "test", 0, 3).has_headroom
    assert Difficulty("t", "test", 1, 3).has_headroom
    assert not Difficulty("t", "test", 3, 3).has_headroom
    assert Difficulty("t", "test", 1, 2).pass_rate == 0.5


def test_screen_aggregates_baseline_runs(tmp_path: Path) -> None:
    tasks = [_task("easy"), _task("hard"), _task("flaky"), _task("stale_ignored")]
    # easy: baseline passes both reps; hard: fails; flaky: 1 of 2.
    _write(tmp_path, "easy", "test", 1, True)
    _write(tmp_path, "easy", "test", 2, True)
    _write(tmp_path, "hard", "test", 1, False)
    _write(tmp_path, "flaky", "test", 1, True)
    _write(tmp_path, "flaky", "test", 2, False)
    # a run whose task is not in the list is stale and ignored
    _write(tmp_path, "gone", "test", 1, False)

    diffs = {d.task_id: d for d in screen(tmp_path, tasks, splits=["test"])}

    assert set(diffs) == {"easy", "hard", "flaky"}  # 'gone' filtered out
    assert diffs["easy"].stratum == SOLVED and not diffs["easy"].has_headroom
    assert diffs["hard"].stratum == HARD and diffs["hard"].has_headroom
    assert diffs["flaky"].passed == 1 and diffs["flaky"].total == 2 and diffs["flaky"].has_headroom


def test_screen_empty_when_no_runs(tmp_path: Path) -> None:
    assert screen(tmp_path, [_task("t")], splits=["test"]) == []
