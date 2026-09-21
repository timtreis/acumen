#!/usr/bin/env python3
"""Turn benchmark run trees into a verdict: is this skill better than no skill, or is it noise?

`acumen report` gives pass rates. This gives the three numbers a rate does not:

* **mean +- spread over drafts** — the same rulebook text drafted twice scores differently, so a
  single draft's rate is a sample of size one (measured on squidpy: sd ~3 of 36 tasks);
* **a paired sign test** against the baseline — 36 tasks with 5-8 flipping between identical
  drafts cannot resolve a 5-task effect, and this says so out loud;
* **hard / easy split** — tasks the baseline already passes are where a skill *regresses*, and
  that is invisible in an overall rate.

Usage (the first arm is the baseline everything is compared against)::

    python analyze.py noskill=runs/lockbox/noskill champion=runs/lockbox/skill_v7

Drafts of one rulebook share a label before a dot, and are pooled::

    python analyze.py noskill=runs/lb/noskill v7.d1=runs/lb/a v7.d2=runs/lb/b v7.d3=runs/lb/c

Runs killed by a platform failure (network drop, rate limit, overload — anything
`acumen.runner.is_transient` recognises) are excluded rather than counted as failures. Other
errors — an agent hitting its turn cap, a tool output overflowing the SDK buffer — are genuine
outcomes of the run and count as failures.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

from acumen.runner import is_transient

Outcomes = dict[str, bool | None]


def read_arm(root: Path) -> Outcomes:
    """Every task's outcome under a run directory; ``None`` where the platform, not the task, failed."""
    files = list(root.glob("*/*/*/rep_*/result.json")) or list(root.glob("*/*/rep_*/result.json"))
    out: Outcomes = {}
    for f in files:
        r = json.loads(f.read_text())
        out[r["task_id"]] = None if voided(r) else bool(r["success"])
    return out


def voided(result: dict) -> bool:
    """A run the platform killed says nothing about the task; one that failed on its own does."""
    return result.get("reason") == "error" and is_transient(result.get("error") or "")


def sign_test(base: Outcomes, arm: Outcomes) -> tuple[int, int, float, int]:
    """Exact two-sided sign test over tasks where the two arms disagree (McNemar, small n)."""
    paired = [k for k in base if base[k] is not None and arm.get(k) is not None]
    gained = sum(1 for k in paired if not base[k] and arm[k])
    lost = sum(1 for k in paired if base[k] and not arm[k])
    n = gained + lost
    if n == 0:
        return gained, lost, 1.0, len(paired)
    tail = sum(comb(n, i) for i in range(min(gained, lost) + 1))
    return gained, lost, min(1.0, 2 * tail / 2**n), len(paired)


def main(argv: list[str]) -> int:
    """Print the verdict for the arms named on the command line."""
    if len(argv) < 3:
        print(__doc__)
        return 2

    arms: dict[str, Outcomes] = {}
    for spec in argv[1:]:
        label, _, path = spec.partition("=")
        if not path:
            print(f"expected label=path, got {spec!r}")
            return 2
        arms[label] = read_arm(Path(path))
        if not arms[label]:
            print(f"no result.json under {path}")
            return 1

    base_label = next(iter(arms))
    base = arms[base_label]

    print("== per arm (voided runs excluded) ==")
    for label, arm in arms.items():
        valid = [v for v in arm.values() if v is not None]
        voided = len(arm) - len(valid)
        note = f"   {voided} runs voided by platform errors" if voided else ""
        print(f"  {label:14s} {sum(valid):3d}/{len(valid):<3d} = {sum(valid) / len(valid):5.1%}{note}")

    # ponytail: drafts are grouped by the label before the dot. One convention, no config.
    groups: dict[str, list[int]] = defaultdict(list)
    for label, arm in arms.items():
        if "." in label:
            valid = [v for v in arm.values() if v is not None]
            if len(valid) == len(arm):  # a partial arm would bias the mean
                groups[label.split(".")[0]].append(sum(valid))
    if groups:
        print("\n== rulebook score = mean over drafts (the unit that matters) ==")
        for name, scores in groups.items():
            spread = f" sd {st.stdev(scores):.1f}" if len(scores) > 1 else ""
            print(f"  {name:14s} drafts {scores} -> mean {st.mean(scores):5.1f}{spread}")
        print("  Draft noise is a per-skill offset: more TASKS do not shrink it, only more DRAFTS do.")

    print(f"\n== paired sign test vs {base_label} ==")
    for label, arm in arms.items():
        if label == base_label:
            continue
        gained, lost, p, n = sign_test(base, arm)
        verdict = "distinguishable" if p < 0.05 else "NOT distinguishable from noise"
        print(f"  {label:14s} +{gained:2d}/-{lost:2d} of {n:3d} paired   p={p:.3f}   {verdict}")

    hard = [k for k, v in base.items() if v is False]
    easy = [k for k, v in base.items() if v is True]
    print(f"\n== hard: {len(hard)} tasks {base_label} fails | easy: {len(easy)} it passes ==")
    for label, arm in arms.items():
        if label == base_label:
            continue
        recovered = sum(1 for k in hard if arm.get(k))
        kept = sum(1 for k in easy if arm.get(k))
        flag = "  <- regressions on tasks the baseline already passes" if kept < len(easy) else ""
        print(f"  {label:14s} recovers {recovered:2d}/{len(hard):<3d} keeps {kept:2d}/{len(easy)}{flag}")
    return 0


def _self_check() -> None:
    """The sign test is the one piece of real logic here, so it gets a runnable check."""
    base = {f"t{i}": i < 5 for i in range(10)}  # 5 pass, 5 fail
    same = dict(base)
    assert sign_test(base, same) == (0, 0, 1.0, 10)
    better = {**base, **{f"t{i}": True for i in range(10)}}  # recovers all 5 failures
    gained, lost, p, n = sign_test(base, better)
    assert (gained, lost, n) == (5, 0, 10) and p < 0.07, (gained, lost, p, n)
    mixed = {**base, "t0": False, "t9": True}  # one lost, one gained -> pure noise
    assert sign_test(base, mixed)[2] == 1.0
    dropped = {**base, "t9": None}
    assert sign_test(base, dropped)[3] == 9
    net = "API Error: Can't reach the API server (ENOTFOUND)"
    assert voided({"reason": "error", "error": net})
    assert not voided({"reason": "error", "error": "Reached maximum number of turns (40)"})
    assert not voided({"reason": "wrong_answer", "error": None})
    print("self-check ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--self-check"]:
        _self_check()
    else:
        raise SystemExit(main(sys.argv))
