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

---

## 2026-08-15 — Target design: continuous rulebook autoresearch loop

**Task.** Establish what the larger task pool is for, and what it implies. Goal: a continuous
loop that optimizes a *rulebook/codebook* for generating `SKILL.md`, scored on a clean
held-out task set, ideally under cross-validation and stratified by task difficulty, with
tasks exhaustively covering the package so the resulting skill is maximally useful yet lean.

**Approach.** Mapped the goal onto the current architecture; identified which pieces exist,
which are new axes, and where the design would silently fail.

**Result.** This is a two-level optimization — acumen today has only the inner level (skill
drafted → benched → improved). The rulebook becomes the optimized artifact and skills become
intermediates, so `DRAFT_PROMPT`/`IMPROVE_PROMPT` must become versioned, content-hashed
`rulebooks/vN/` mirroring `skills.py`.

Missing, in dependency order:
1. Coverage-driven task generation at volume (prerequisite for all of it).
2. A difficulty signal — none exists; the only honest source is baseline pass rate, so a
   screening bench becomes mandatory. Difficulty is model-dependent, so the reference model
   must be fixed and recorded.
3. CV over tasks — a genuinely new axis. Today's `train`/`test` is *within-task variants*
   (`tasks.py:27`), not a task partition; the two are orthogonal and both are wanted.
   `SPLITS` is a hardcoded path component (`paths.py:17`); `collect_train_runs`
   (`improve.py:137`) has no task filter; the isolation hook blocks `runs/*/test/` only.
4. Coverage as a *measured* quantity. Denominator: squidpy documents 68 public symbols
   (34 `gr`, 15 `datasets`, 10 `pl`, 4 `im`, 3 `read`, 2 `tl`) → ~43 analysis-bearing after
   dropping datasets and plotting. Proposal: instrument the ground-truth `script.py`
   execution to record which symbols were actually called — verified coverage, and uncovered
   symbols become the generation queue. Same "evidence not assumption" pattern as
   `skill_loaded`.

Traps identified:
- **Selection leakage.** Scoring rulebook v1..vN against the same folds turns held-out into
  training-by-selection. Needs nested CV or a structurally unreadable lockbox.
- **Cost.** ~150 tasks × 5 folds at the scaffolded config ≈ 2,700 runs/iteration (~$540).
  Tractable form: cheap inner loop (1 model, 1 rep, held-out only ≈ 150 runs, ~$30) plus rare
  full-config confirmation. Eval config must become a first-class knob.
- **Dataset re-download** moves from inefficiency to blocker at this run count; a warm shared
  cache is required without loosening the env scrub.
- "Lean" is unmeasured — no skill-size metric exists; it needs a Pareto axis against success
  rate (frontier machinery exists at `report.py:593,631`, currently cost vs success).
- A continuous unattended loop deliberately inverts the shipped skill's "never chain commands
  unattended" stance, making stopping rules and hard budget caps load-bearing.

**Decisions.** Rulebook is **squidpy-specific** (CV partitions squidpy tasks, not packages).
Budget: **assume infinite tokens** — so wall-clock, not dollars, is the binding constraint
(~2,700 runs/iteration at `max_concurrency: 4` is ~30h, and squidpy's permutation-heavy
analyses are CPU-bound locally). Infinite budget also makes selection leakage *worse*, since
many more iterations degrade the held-out set faster — the lockbox is now essential, and
dataset caching plus concurrency move up the build order.

**Build order agreed.** P0 clone submodules → P1 sharded task generation → P2 coverage
measurement → P3 warm dataset cache → P4 difficulty strata → P5 CV axis + lockbox →
P6 rulebook artifact → P7 the loop.

---

## 2026-08-15 — P0: check out target submodules

**Task.** Make squidpy's tutorials visible to the drafting and task-generation agents.

**Approach.** Added a `submodules: bool = True` config key and made `env.py:_clone` run
`git submodule update --init --recursive` *after* the ref checkout, so submodules land on the
commit the ref pins. A declared-but-unfetchable submodule raises `EnvError` naming the opt-out
rather than leaving an empty directory — an agent cannot tell that apart from "this package
has no tutorials", which is the exact silent failure being fixed. `submodules` is recorded in
the cache ready-marker and a mismatch forces a rebuild, so flipping it cannot keep serving the
submodule-less tree it was first built with.

**Result.** Verified against real squidpy: **50 notebooks** now checked out under
`docs/notebooks` (20 tutorials, 29 examples, 1 deprecated) where the previous code produced 0.
Three tests added (default + validation, command order and opt-out, hard failure on an
unfetchable submodule); full suite 111 passed, ruff clean.

Note for existing users: cached targets built before this change are invalidated by the
marker check and will re-clone on next use.

---

## 2026-08-18 — P1: sharded task generation (one agent per notebook)

**Task.** Break the single-context task generator into one agent per notebook, so coverage is
the union of many small agents instead of one agent's stamina — with per-shard failure
isolation, free resume, and a validated merge. **Decision:** shard = notebook (50 shards, option
A). B's only advantage was download cost, which the committed P3 warm cache retires regardless of
shard key; A keeps the natural analysis boundary, small per-shard context, and the clean
per-notebook mapping the taskgen prompt is already built around.

**Approach.** Enumerate notebooks *in the harness* (`notebook_shards`: `rglob("*.ipynb")` minus
checkpoint copies), not inside an agent, so fan-out/resume/merge are code-driven. Extracted the
isolation-critical agent core (`_run_generation_agent` — scrubbed env, skill-bias guard, filtered
`add_dirs`, query loop) so the whole-package and per-notebook paths share exactly one copy of the
security-sensitive part and can't drift; the whole-package `generate_tasks` now calls it too.
`generate_tasks_sharded` builds **one** read-only filtered source copy shared across all shards
(isolation is one `copytree`, not 50), then fans out `taskgen_shard_prompt` agents under a
semaphore. Each writes its own validated `shards_dir/<slug>.yaml`; a shard that raises is recorded
`failed` and skipped rather than taking the pass down (mirrors `run_matrix`); a shard whose file
already parses is reused (resume by file presence, like the runner's `result.json`). `merge_shards`
namespaces every task id by shard slug (`<slug>__<id>`) so the union is unique and traceable, then
round-trips through the strict loader before writing `tasks.yaml`.

The prompt stays one template: `TASKGEN_PROMPT` grew a `{scope}` + `{coverage_check}` seam, and
`taskgen_shard_prompt` fills the per-notebook scope while `taskgen_prompt` fills the whole-package
one — so the task-writing rules (ground-truth-by-execution, output schema, train/test variants)
have a single source of truth and only the coverage scope differs. CLI: `acumen tasks
--per-notebook [--shards-dir DIR]`, default off (non-breaking; the whole-package path is unchanged
and the P7 loop will call the API directly). `force` governs only `out_path`; regenerating a shard
means deleting its file — the deliberate, inspectable knob.

**Result.** New `tests/test_taskgen.py` (11 tests) covers the pure seams (enumeration/checkpoint
skip, slug disambiguation of same-named notebooks in different galleries, id namespacing, merge
uniqueness) and the fan-out with `_run_generation_agent` monkeypatched — merge-all, failed-shard
isolation, cached-shard resume (agent never invoked), no-notebooks and existing-out errors, and
that each shard is handed its own notebook (guards the closure late-binding trap). Full suite
**122 passed**, ruff 0.16.1 clean. No real agent run yet — that needs auth + long wall-clock
against prepared squidpy; the orchestration is verified offline.

Deferred to later steps: `--max-concurrency` override on `tasks` (uses `cfg.max_concurrency`);
the warm dataset cache that makes 50 shards cheap (P3); tuning tasks-per-notebook (prompt asks for
"one to three, one per distinct analysis" — no knob until evidence says one is needed).
