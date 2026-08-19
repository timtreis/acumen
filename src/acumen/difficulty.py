"""Task difficulty screening — baseline pass rate as the only honest difficulty signal.

A skill (and the rulebook that writes it) can only be shown to help on tasks the **baseline** — an
agent with no skill — does not already solve. There is no intrinsic difficulty label on a task, and
guessing one is dishonest, so we *measure* it: run the baseline (`noskill`) arm and read the per-task
pass rate. A task the baseline always passes has no headroom (a skill cannot improve a perfect
score); a task the baseline always fails, or passes only sometimes, is where a skill's guidance can
move the number — those are the tasks the loop must be scored on.

Difficulty is **model-dependent** — an easy task for one model is hard for another — so screening is
always relative to the model(s) that produced the baseline runs, and the caller is responsible for
fixing and recording that reference model (it is stamped in each ``result.json``).

This module only *reads* baseline runs; produce them first with ``acumen bench --no-skill``. Kept
read-only and pure so it composes with the existing run tree and resume, rather than launching a
second, subtly different benchmark path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen.paths import NOSKILL_ARM, RESULT_FILE, SPLITS, Split, parse_run_dir
from acumen.tasks import Task

#: Difficulty strata, by baseline pass rate. ``solved`` has no headroom for the loop; ``hard`` and
#: ``flaky`` do (a skill can move a score the baseline does not already max out).
SOLVED = "solved"
FLAKY = "flaky"
HARD = "hard"
UNSCREENED = "unscreened"


@dataclass(frozen=True)
class Difficulty:
    """The baseline outcome for one (task, split): how often the no-skill arm passed."""

    task_id: str
    split: Split
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        """Baseline pass rate in ``[0, 1]``; ``0.0`` when nothing was screened."""
        return self.passed / self.total if self.total else 0.0

    @property
    def stratum(self) -> str:
        """The difficulty bucket: ``solved`` / ``flaky`` / ``hard`` / ``unscreened``."""
        if self.total == 0:
            return UNSCREENED
        if self.passed == 0:
            return HARD
        if self.passed == self.total:
            return SOLVED
        return FLAKY

    @property
    def has_headroom(self) -> bool:
        """Whether a skill could improve on the baseline here (baseline does not already pass all)."""
        return self.total > 0 and self.passed < self.total


def screen(
    runs_root: Path,
    tasks: Sequence[Task],
    *,
    arm: str = NOSKILL_ARM,
    splits: Sequence[Split] = SPLITS,
) -> list[Difficulty]:
    """Read the baseline arm's runs and tally per-(task, split) pass rate.

    Parameters
    ----------
    runs_root
        The ``runs/`` root.
    tasks
        The tasks to screen; a run for a task not in this list is ignored (stale).
    arm
        The arm to read as the baseline — ``"noskill"`` by default. Pass a skill arm to screen
        that skill's difficulty instead.
    splits
        Which splits to screen.

    Returns
    -------
    One :class:`Difficulty` per (task, split) that has at least one baseline run, sorted by
    task id then split. Pooled over every model and replicate found — if you screened more than
    one model, split them yourself before calling to keep the per-model reading honest.
    """
    known = {task.id for task in tasks}
    tally: dict[tuple[str, Split], tuple[int, int]] = {}
    for split in splits:
        split_root = runs_root / arm / split
        if not split_root.is_dir():
            continue
        for result_path in sorted(split_root.rglob(RESULT_FILE)):
            key = parse_run_dir(runs_root, result_path.parent)
            if key.task_id not in known:
                continue
            try:
                data = json.loads(result_path.read_text())
            except (OSError, ValueError):
                continue
            passed, total = tally.get((key.task_id, split), (0, 0))
            tally[(key.task_id, split)] = (passed + bool(data.get("success")), total + 1)
    return [
        Difficulty(task_id=task_id, split=split, passed=passed, total=total)
        for (task_id, split), (passed, total) in sorted(tally.items())
    ]
