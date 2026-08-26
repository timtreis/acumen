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

---

## 2026-08-19 — Spike: crude end-to-end rulebook loop (P6+P7 prototype)

**Task.** Before investing in P2–P5, prototype the two-level optimizer to answer one feasibility
question: *does optimizing the rulebook (the SKILL-generating instructions) — not a skill directly
— actually move a held-out score, and is the failure signal legible enough to drive the next
rulebook version?* Classified as a **spike**: throwaway-ish, runnable on a Claude subscription so
the user (or anyone) can run it, informs the real plan.

**Key finding, surfaced building it.** acumen could **not** already run the loop on a subscription:
`_cmd_bench` hardcodes `resolve_auth_mode("api", allow_session=False)` because bench records real
per-run `cost_usd`, so a subscription user is refused. The loop's inner scoring *is* bench. The fix
was lazy: the underlying `run_matrix`/`run_once` already accept `auth_mode="session"` — only the
bench *CLI preflight* forbids it. The loop is new code, so it calls `run_matrix(auth_mode="session")`
directly and scores **pass rate, not dollars**; the existing `bench` command's API-only cost
semantics are untouched. This is the kind of structural snag the spike existed to find, and it
would have blocked P7 regardless.

**Design realization that shrank the prototype.** "Held-out" in the crude version is the *existing
within-task test split* (`tasks.py` train/test variants), not a task partition. So: rulebook
improvement is driven by **train**-split failure evidence, the rulebook is scored on **test**-split
pass rate, and leakage protection comes **free** from the same `runs/*/test/` guard the skill
improver already uses. No new slicing, and I reuse `collect_train_runs` / `_write_material` /
`make_test_guard` wholesale. Honest per the target design's own note that within-task train/test is
orthogonal to the eventual CV-over-tasks axis (P5 adds the task partition).

**Approach.** `rulebooks.py` (crude P6): the rulebook is the draft-prompt *template text*, versioned
at `rulebooks/vN/rulebook.md`; v1 is seeded verbatim from `DRAFT_PROMPT` (so iteration 0 reproduces
today's drafting exactly — verified byte-identical). `validate_rulebook` guards the two ways the
outer agent can break the template (a stray `{placeholder}` the drafter can't fill; a dropped
`{out}`/`{skill_name}`). One seam threads it through: `draft_prompt(template=...)` →
`draft_skill(rulebook=...)`, both defaulting to the built-in so plain `acumen draft` is unchanged.
`loop.py` (crude P7): `improve_rulebook` mirrors `improve_skill` one level up (edits the rulebook
from train evidence, same held-out guard); `run_iteration` drives seed v1 → draft → bench both
splits → score test → improve rulebook → draft v2 → bench test → score, reporting P1 vs P2 + the
v1→v2 rulebook diff + the improve rationale. Resumable by file presence at every step (drafted
skills, bench runs, and the improved rulebook all skip if already on disk). New outer-improve prompt
`RULEBOOK_IMPROVE_PROMPT` hammers "improve the methodology, never the instance" and "preserve every
placeholder." Exposed as `acumen loop` (subscription by default), which prints a blunt prototype
caveat: single iteration, within-task held-out, not protected against selection leakage — it
measures whether the loop *moves a score*, not a trustworthy final number.

**Result.** New `tests/test_loop.py` (7 tests): rulebook seeding/idempotence/immutability/validation,
the draft-prompt transparency of v1, the disk scorer, and `run_iteration` end-to-end with all three
agent boundaries (`draft_skill`, `run_matrix`, `improve_rulebook`) monkeypatched — a scripted +1
held-out move, exact version lockstep and cost accounting, and a resumed run that respawns **zero**
agents yet reproduces the scores from disk. Full suite **129 passed**, ruff 0.16.1 clean. Fixed a
`seed_default` bug found by the resume test (it returned the latest version, so a resumed loop
mistook v2 for its baseline; now always anchors on v1).

**Not yet run for real.** The feasibility *answer* needs a subscription run against prepared squidpy
(`acumen loop`, tens of minutes). The orchestration is verified offline; the real signal — did
held-out move, was the rulebook diff sensible — comes from that run and will set the P2–P5 plan.
Reused private helpers `_write_material`/`_read_rationale` from `improve.py` (same package, prototype
scope). When P5 lands (CV + lockbox), the loop's control flow gets restructured; this is the seed,
not the final loop.

---

## 2026-08-19 — Cost is not an optimization metric; unblock `bench` on a subscription

**Task.** User steering: dollar/token cost is not a metric to optimize for or against. Budget is
effectively infinite; the binding constraints stay wall-clock and selection leakage.

**Change.** The one place this was load-bearing in code was `bench`: it forced API billing purely to
record real per-run `cost_usd`. Removed the `allow_session` parameter from `resolve_auth_mode`
entirely (it existed only to model bench's API-only stance, and became dead once cost stopped
mattering), gave `bench` the same `--auth {auto,session,api}` flag every other command has, and
updated the docstrings/comments that claimed bench must bill the API. A subscription `bench` now
runs and simply records no meaningful cost. This makes **all** of acumen subscription-runnable,
consistent with the loop, which already scored pass rate rather than dollars. Merged the two
`resolve_auth_mode` tests into one (the bench-specific case is gone). Full suite **128 passed**,
ruff clean. Saved the steering to project memory (`cost-not-an-optimization-metric`).

**Implication still open (not done):** the report's "lean yet useful" Pareto frontier is currently
cost-vs-success (`report.py`); under this steering the second axis should become **skill size**, not
cost. Deferred to the leanness work — flagged, not yet built.

---

## 2026-08-19 — First real end-to-end loop run on squidpy (subscription)

**Task.** Run the crude loop once against real squidpy on the Claude subscription — the feasibility
answer the spike was built for.

**Setup.** Workspace at `~/acumen-squidpy` (config → scverse/squidpy, sonnet, 1 rep). Auth: minted
a `CLAUDE_CODE_OAUTH_TOKEN` via `claude setup-token` (the machine's login is in the macOS Keychain,
which acumen can't read — token file is the bridge). Prepared the target (no auth), then **authored
one task by hand from executed ground truth** rather than trust the unproven task-gen agent:
"most spatially variable gene by Moran's I", train `slideseqv2 → Ttr`, held-out `merfish → Nnat`,
both computed in the target venv and checked for normalization-robustness.

**What happened.** The loop ran the whole two-level machinery to completion on the subscription:
draft skill from rulebook v1 → bench (train+test, `auth=session`) → improve rulebook to v2 → draft
skill v2 → bench (test) → report. Held-out `1/1 → 1/1`, **moved +0**.

**The signal — strongly positive on the core question.** The improve agent read the one failing
train run and made a *sensible, general, non-overfit* rulebook edit. It correctly diagnosed a subtle
failure (the drafted skill flagged an expensive full-gene Moran's call as "expensive… if runtime
matters" but gave no lever to bound it; the sandbox agent ran it in the background and stalled its
whole turn budget polling for a result that never came → no `answer.md`) and generalized the fix:
a new drafting-rulebook bullet — *"Flag what runs long, and say how to bound it… name the concrete
knob (subset/threshold/parallelism/cheaper mode), don't just call it expensive."* It explicitly
refused to overfit ("not specific to Moran's I… squidpy has permutation tests, `sepal`, image
processing with the same shape"). So the spike's real question — *is the failure signal legible
enough to drive a good rulebook change?* — is answered **yes**.

**What it did NOT show, and why.** No score movement, for two calibration reasons the run itself
surfaced (both predicted by the target design):
1. **Difficulty (P4) is a precondition, not a nicety.** The baseline skill v1 *already passed* the
   held-out merfish task → held-out at ceiling, zero headroom. Without tasks the baseline fails, the
   loop cannot demonstrate movement.
2. **Benchmark-harness weight limit.** A heavy task (slideseqv2, 4000 genes) makes the sandbox agent
   background its work and idle waiting for a notification that never arrives in a headless run →
   spurious `no_answer_file`. Keep tasks inline-light, or harden the sandbox against backgrounding.
   Also the crude loop only re-benches v2 on the *test* split, so v2's targeted fix (bounding heavy
   calls) was never exercised on the heavy case it addresses — a real-loop design note.

**Verdict.** The two-level optimizer concept works end-to-end and produces legible, generalizable
rulebook improvements from real evidence — the bet pays off. The blockers to a *measurable* result
are exactly P4 (difficulty-calibrated tasks with baseline headroom) and honest re-benching of the
improved version on the cases it targets. Findings saved to memory
(`bench-agent-backgrounds-heavy-tasks`, `task-supply-is-large-scale-mining`). Artifacts left in
`~/acumen-squidpy` (rulebooks/v1,v2, skills/v1,v2, runs/, loop.log, logs/).

---

## 2026-08-19 — Critical path, steps 1–2: reliable bench + first proven task-gen

**Plan.** Chose the "critical path to a measurable loop": (1) stop the bench sandbox stranding a run
by backgrounding; (2) prove P1 task-gen actually works on a real agent; (3) build P4 difficulty
screening and rerun the loop on a baseline-failing held-out.

**Step 1 — reliable bench (done, shipped).** Added `find_background_use` + `make_sync_guard`
(`runner.py`), a `PreToolUse` hook that denies `run_in_background` and monitor-style tools in the
sandbox, identical across arms so baseline parity holds; plus a synchronous-execution line in the
harness preamble. Unit-tested. This converts a stranded run into either an inline completion or an
honest cap breach.

**Step 2 — task-gen proven (done).** Added a small `--notebook SUBSTR` filter to the sharded path
(`generate_tasks_sharded` + CLI) so a run can target a handful of notebooks. Ran the **first real
P1 task-gen agent** against squidpy's `examples/graph/compute_centrality_scores` on the
subscription. Result: **3 well-formed tasks** ($0.80), each an imc(train)/seqfish(test) pair over a
distinct centrality measure (degree / closeness / clustering), one-paragraph goals with precise
categorical outputs. Independently recomputed all six ground-truth answers with squidpy — **6/6
correct** (`CK low HR low tumor cell`, `apoptotic tumor cell`, `endothelial`; `Cardiomyocytes`,
`Low quality`, `Erythroid`). So P1 generates trustworthy tasks with real executed ground truth, not
just valid-looking YAML. Minor quality note: prompts are a touch recipe-ish ("build a spatial
neighbor graph and compute…") vs. the pure lazy-goal ideal — a later rulebook/prompt refinement, not
a blocker. These centrality tasks are also promising *calibrated* candidates: exact-match on a
cell-type string like "CK low HR low tumor cell" is unforgiving, so the baseline likely fails them
(headroom) — step 3 will confirm.

---

## 2026-08-19 — Critical path, step 3: difficulty screening + a sharper loop metric

**Built P4 screening (`difficulty.py` + `acumen screen`).** `screen()` reads the baseline
(`noskill`) arm's runs and tallies per-(task,split) pass rate into strata (solved / flaky / hard /
unscreened); `has_headroom` marks tasks the baseline does not already pass — the only ones a skill
can be shown to help on. Read-only over the run tree, per reference model, unit-tested.

**Screening the generated centrality tasks was itself the finding.** Benched the baseline on all 3
generated tasks:
- **sonnet baseline: 6/6 passed** → `screen` reported *no task has headroom*. sonnet solves
  straightforward squidpy analyses with no skill.
- Switched the *reference/bench model* to **haiku** (difficulty is model-dependent — the design
  said so; kept draft/improve on sonnet so authoring stays strong). **haiku baseline: 5/6**, failing
  only `closeness_most_central[test]` (seqfish). `screen` cleanly flagged that one as `hard` /
  headroom. Note: that run took ~14 min because closeness centrality on a 19k-cell graph is
  genuinely compute-heavy — a task-weight signal for the pool (favor cheap analyses).

**Sharper loop metric (`noskill_score`).** The crude loop only compared rulebook v1→v2 held-out,
which needs skill_v1 *itself* to fail — a demanding bar that the "does the skill help at all"
question doesn't. Added `noskill_score`: the loop now reads the baseline arm's held-out pass rate
from its runs tree (reusing a prior `screen`) and reports **noskill → skill v1 → skill v2** with both
deltas (skill-vs-noskill, and rulebook-v1-vs-v2). This is the more fundamental signal and far
easier to observe than rulebook self-improvement.

**Measured loop run on the headroom task** (`closeness_most_central`, haiku agent, sonnet authoring;
one mid-run failure when the Mac slept — reran under `caffeinate -i`, resumed cleanly). Result:
**noskill 0/1 → skill v1 0/1 → skill v2 0/1**, no movement. But the *why* is two calibration
problems, not a loop failure:

1. **The skill never loaded** — `skill_loaded=False` on every arm including v2. haiku does not
   reliably load skills, so the whole skill mechanism is neutralized; no rulebook change can help a
   model that won't open the skill. The weak-model shortcut for headroom backfires: weak models
   fail tasks *and* don't load skills. A demonstrable loop needs a **skill-loading** model (sonnet),
   which means tasks hard *for sonnet*, not merely for haiku.
2. **The generated task's answer is adversarial.** Ground truth `Low quality` is a QC-artifact group
   that tops raw closeness; every agent (reasonably) reported a real cell type
   (`Haematoendothelial progenitors`, `Cardiomyocytes`) and was marked wrong. Task-gen's "run it,
   take the top" verification is insufficient — it must reject answers a thoughtful analyst would
   exclude. A task-quality lever for the generation rulebook.

The improve agent again reasoned well: v2's rationale correctly diagnosed a *description-coverage*
gap (the v1 skill's description omitted centrality, so it couldn't load for a centrality task) and
generalized the fix (enumerate the full public API, cross-check every capability against the
description). The intelligence is sound; the bottleneck is decisively **task calibration** —
hard-for-a-loading-model tasks with defensible answers — which is where P4 / generation effort must
go, not the loop code. Saved to memory (`loop-needs-hard-tasks-on-a-loading-model`).

**Critical-path status:** machinery all built and proven (reliable bench, real task-gen, difficulty
screening, noskill→skill loop metric); the measured run has pinned the remaining blocker to task
calibration + reference-model choice rather than any code gap.

---

## 2026-08-19 — Skill-load rate as a first-class loop metric

**Task (user steering).** The measured run failed invisibly on pass rate: every arm scored 0/1, but
the reason (skill never loaded) was only findable by digging into result.json. Make skill-loaded a
success metric the loop tracks and reports.

**Change.** Extended `Score` (loop.py) with a `loaded` count and `load_rate`; `score()` now tallies
`skill_loaded` alongside `success` (they are orthogonal — a run can load and still fail, or pass
without a skill). Added `LoopResult.load_moved` (v1→v2 load delta) beside `moved`. The CLI loop
report now prints a **held-out PASS** line and a **held-out LOAD** line (noskill → skill v1 → skill
v2) and both deltas. Verified on the real closeness run (fully resumed, $0.00): PASS 0→0→0, LOAD
0→0→0 — now visible that v2's description fix did not lift loading on haiku (which doesn't load
skills), which is exactly why no rulebook change could move the score. Full suite 132 passed.

---

## 2026-08-23 — P2 coverage: inventory + reference scan (P2a)

**Task (user).** Fully build P2 (coverage measurement) and P3 (warm dataset cache), after a
design comparison of built-vs-scope. Approved the plan: AST-over-persisted-scripts attribution
(not a runtime tracer), config-declared shared cache symlinks, pre-warm to kill the download race.

**P2a — `coverage.py` (new) + `tests/test_coverage.py`.** Two pure pieces, no agent, no download:
- `build_inventory(pkg)` introspects the *installed* package: walks the public module tree
  (`pkgutil.walk_packages`, skipping any `_`-prefixed component), collects public functions/classes
  whose `__module__` is in-package (excludes dependency re-exports), and records the **public
  attribute path** a user writes (`squidpy.gr.spatial_neighbors`), not the private definition
  module. JSON round-trips (`Inventory.write/read`) so the import-heavy scan runs once per version.
- `scan_references(script, pkg)` resolves `import x as y` / `from x import f` against the package by
  AST and returns the package-qualified names a script statically references — maximal attribute
  chains only (the inner `sq.gr` module is passed through, not counted), conservative on anything
  unresolvable. `measure_coverage` unions per-task refs ∩ inventory → `covered`; `uncovered` is the
  generation queue.

**Result.** 10 tests pass; ruff clean. Smoke on real squidpy: 99 public symbols across
gr/pl/im/tl/read/datasets (+experimental). Denominator is the honest public surface (includes some
builder plumbing like `gr.neighbors.*` and `assert_positive` — curation is target-specific tuning,
deferred, not a mechanism gap). Next: P2b — persist one ground-truth script per task from task-gen.

---

## 2026-08-23 — P2 coverage: persist ground-truth scripts + `acumen coverage` (P2b/P2c)

**P2b — attribution input.** Task-gen now persists one confirmation script per task so coverage has
something to statically analyse. Prompt (`prompts.py`): added a `{scripts_dir}` placeholder and told
the agent to save each task's exact script to `{out_dir}/scripts/<id>.py` (kept only for coverage;
the benchmarked agent never sees them). `taskgen.py`: `_harvest_scripts` reads `work/scripts/*.py`
keyed by task id (drops strays); `write_scripts` persists them; `_run_generation_agent` now returns
`(tasks, scripts, result)`. Non-sharded writes `out/scripts/`; each shard caches to
`shards_dir/<slug>/` and `merge_shards` collects them under the SAME namespaced id the tasks get
(`<slug>__<id>.py`) so coverage keys line up with tasks.yaml ids. **Design call:** a missing/unsaved
script is NOT a hard error — the script is a coverage-only artifact and discarding a valid task set
(a full agent re-run) over it is the wrong trade; coverage degrades gracefully and the gap is
surfaced instead.

**P2c — `acumen coverage`.** Builds the inventory by shelling out to the **target venv** interpreter
(`coverage.inventory_in_venv`) — the package imports only there, and the bootstrap loads coverage.py
by path to dodge `acumen/__init__`'s SDK import; stdout is redirected during import so a package
banner can't corrupt the JSON. Then reads `scripts/` beside tasks.yaml, measures, and prints
coverage % + the uncovered generation queue (`--queue` emits bare names for `--feedback`). Names
tasks with no script.

**Result.** +13 tests (145 total), ruff clean. Real squidpy: inventory = 99 public symbols;
`acumen coverage` runs end-to-end on the workspace (0/99 — the 3 pre-P2b tasks have no scripts,
correctly flagged). Dropped ONE real hand-written squidpy script into a scripts dir → 4/99, with
`datasets.merfish`, `gr.spatial_neighbors`, `gr.centrality_scores`, `pl.centrality_scores` all
resolved correctly — the AST scan matches real squidpy call style. Not yet proven: a *live* task-gen
agent actually saving the scripts (mock + prompt-render tests cover it structurally; a real run is
an expensive separate step). Next: P3 warm cache.

---

## 2026-08-26 — Master build plan + P3 warm dataset cache

**Plan (user: "build EVERYTHING else").** Wrote `tasks/build-plan.md`: sequential build, one module
at a time — P3 warm cache → P6-full rulebook → leanness axis → P4-full strata → task mining
(GitHub + web, full) → P5 CV+lockbox → P7-full loop. Decisions: fable authors the mechanical
modules, the CV/lockbox core stays on a strong model. Key ordering insight: P5 over 3 tasks is
meaningless, so the mining corpus comes before P5.

**P3 — mechanism.** Root cause confirmed in the source: squidpy writes to
`scanpy.settings.datasetdir`, default `data` — **cwd-relative**, so every throwaway sandbox
re-downloaded. Fix is a filesystem redirect of acumen-owned paths, not an env change:
- `config.dataset_cache_dirs` (validated single path components; squidpy: `[data, cache]`).
- `Target.datasets_dir` = `<cache entry>/datasets` (shares the venv's cache key; dropped on
  `--refresh-target`).
- `sandbox.link_dataset_cache`: symlink `<root>/<name>` → shared dir, both arms identically
  (parity), env scrub / HOME / XDG untouched. Threaded cli → run_matrix → run_once → sandbox
  exactly like `env_passthrough`.
- `warm.py`: lifts `sq.datasets.X(<literals>)` calls out of the persisted P2 scripts by AST
  (reuses coverage's alias resolver — extracted `collect_aliases`/`resolve_parts`/`attr_chain` so
  there is ONE resolver), dedupes, and runs each once **sequentially** in the target venv with the
  shared dir as cwd. Kills the concurrent first-download race. Computed args are skipped, never
  guessed. `acumen warm`; auto-warm before the matrix in `bench`/`loop` (`--no-warm` to skip),
  gated on `dataset_cache_dirs` being set (otherwise sandboxes can't see the shared dir anyway).

**Result.** +7 tests (152 total) incl. a real subprocess warm against a fake package; ruff clean.
Real squidpy: `acumen warm` downloaded `merfish.h5ad` (49 MB) into
`~/.cache/acumen/squidpy-*/datasets/data/anndata/` in 45 s; re-run 9.8 s with no download. Not yet
proven: a full bench pass reading through the symlink (unit test covers the link; the loader's
cwd-relative resolution is what the real warm proved). Next: P6-full rulebook artifact.
