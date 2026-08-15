# Diary

A running log of work on this fork. One entry per task: what was asked, how it was
approached, and a brief result. Newest entries at the bottom.

---

## 2026-08-15 — Repo overview

**Task.** Get an overview of what acumen is and does.

**Approach.** Read `README.md`, `pyproject.toml`, and the modules under `src/acumen/`
(CLI, config, tasks, bench, runner, sandbox, grade, skills, paths), plus the skill acumen
ships for itself (`src/acumen/_skills/data/SKILL.md`).

**Result.** acumen builds, benchmarks, and optimizes agentic Skills for Python packages.
Loop: `init` → `tasks` → `draft` → `bench` → `improve` → `report` → `ship`. Load-bearing
design points: baseline parity (arms differ only by the presence of a skill dir in the
sandbox), test-split isolation enforced structurally *and* by a `PreToolUse` hook, immutable
content-hashed skill versions, exact-string grading with `format_error` split out from
`wrong_answer`, and resume keyed on `result.json` existing. ~8.1k lines, 89 tests.

---

## 2026-08-15 — Task-generation deepdive

**Task.** Understand how `acumen tasks` generates tasks, with a view to producing a much
larger task pool, targeting Squidpy.

**Approach.** Read `taskgen.py`, `prompts.py:TASKGEN_PROMPT`, `env.py` (target prep and env
scrub), and `references/setup.md`; traced the removal of `--append` (`42eac90`). Then cloned
`scverse/squidpy` and `scverse/squidpy-tutorials` to check the assumptions against the actual
target.

**Result.** Generation is a *single* agent in a single context: enumerate tutorials → read
source → execute every ground-truth pipeline → emit one `tasks.yaml`, validated all-or-nothing
and written only on success (the temp dir is deleted in a `finally`, so a cap breach or error
loses everything). No fan-out, no checkpointing, no append (removed deliberately: the
generator is blind to existing tasks, so appending risked semantic duplicates).

Scaling blockers: single-context serialization, total failure on any error, strict
whole-file validation, the shard key (tutorial) being discovered *inside* the agent, ground
truth recomputed per task, and no quality screening.

Squidpy-specific findings:
- `docs/notebooks` is a **git submodule** (`scverse/squidpy-tutorials`) and `env.py:_clone`
  does not recurse submodules — verified empty in a fresh clone. The prompt's central
  "one task per tutorial" instruction therefore binds to nothing and silently falls back to
  inferring from source. The submodule holds **50 notebooks** (20 tutorials, 9 `examples/graph`,
  15 `examples/image`, 3 plotting, 2 tools; 146 MB) — a natural shard list of about the right
  cardinality for the pool we want.
- Throwaway `HOME`/`XDG_CACHE_HOME` plus squidpy's `settings.datasetdir` mean every taskgen
  agent *and* every bench run re-downloads its dataset (16 registered, several large).
- Pool size couples hard to bench spend: the scaffolded config is 18 agent runs per task per
  arm, so 100 tasks is ~1,800 runs per arm.

Proposed direction (not yet implemented): clone submodules; enumerate notebooks in the
harness and fan out one agent per shard behind the existing `asyncio.Semaphore` pattern, each
writing its own validated shard file (per-shard failure, free resume, structural dedup); ask
for k tasks per shard off one prepared object; a shared warm dataset cache; and a merge step
plus optional cheap baseline screening.
