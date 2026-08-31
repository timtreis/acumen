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

---

## 2026-08-26 — P6-full: the rulebook as a real artifact

**Change.** `rulebooks.py` now mirrors `skills.py` for real instead of "a versioned file":
- **Content-hashed** — `rulebook_hash` is `skills.skill_hash` over the same notion of content files,
  so the two artifact kinds share one hash definition; same text ⇒ same hash regardless of version
  name, so a score attaches to the *text*.
- **Immutable, enforced both ways** — `write_rulebook` refuses if the version *directory* exists at
  all (a half-written version is a collision too) and validates the template before touching disk;
  `load_rulebook` re-hashes and compares against `meta.json` → an in-place edit raises
  ("modified since it was written") instead of being silently scored under the old name.
- **Provenanced** — `meta.json` (version, parent, rationale, hash, feedback) written by the SAME
  `skills.write_meta`, deliberately: one provenance schema, one reader, one chain from a shipped
  skill back through the rulebook that drafted it. `seed_default` records its origin; the loop's
  `rationale.md` side-file is gone (rationale lives in meta). `load_rulebook` returns a `Rulebook`
  (`.text/.hash/.path`); a pre-provenance dir (no meta) still loads, so the existing squidpy
  workspace isn't broken.
- `loop.py` migrated (`.text`, provenance-writing `write_rulebook`, resume reads meta).

**Result.** +8 tests in `tests/test_rulebooks.py` (160 total): provenance chain, same-text-same-hash,
tamper detection, dir-level immutability, validate-before-write, seed provenance. Ruff clean. Next:
leanness axis (size, not cost) in report.py.

---

## 2026-08-26 — Leanness axis = skill size, not cost (report restructure)

**Change.** The user's steer (cost is not a metric acumen optimizes) reaches the report:
- `skills.skill_size(dir)` = bytes of content files (SKILL.md + references; meta excluded, like the
  hash). `Skill.size`; bench records `skill_bytes` in every `result.json` (0 for the baseline —
  "no skill" genuinely weighs nothing, which is what anchors the frontier at the origin).
- `load_results(skills_root=)` recovers `skill_bytes` for pre-field runs from the skill tree (size is
  a property of the version dir); unrecoverable → NaN → left off the axis, never guessed.
- **Trade-off figure**: x = skill size (formatted B/KB), frontier over (size, rate). "Size vs. success".
- **Significance section — the real design change.** The old test was two-axis dominance ("cheaper
  AND better", intersection–union p). With size, the second axis has zero sampling variance and the
  baseline is the leanest thing possible by definition, so "leaner and better than baseline" is
  vacuous. Restructured honestly: the paired, task-clustered bootstrap + Holm now tests **rate only**
  (`_rate_p`); size is reported as the price paid; **On frontier** (`_frontier_probability` with a
  fixed per-arm size axis) says whether that price was justified — how often no smaller skill matches
  the rate. `p_cost`/`d_cost`/`_dominance_p` deleted. Prose rewritten. `_arm_sizes` reads the test
  split only.
- Cost stays as *data* (grid column, runs table) — it's informational, just not an objective.

**Result.** 160 tests pass (rewrote the trade-off + significance tests for size; a subtle one: a
better-but-bigger skill's frontier share is ~0.98, not 1.0, because rare resamples drawing only the
tasks the baseline also passes tie it at 100% and the smaller baseline wins — correct behaviour,
the share is robustness not certainty). Real report on the squidpy workspace: baseline 0 B, v1
21.5 KB, v2 34.2 KB via the skills-tree fallback — and the improve step made v2 *bigger* for no
pass gain, which is precisely what this axis is for. Possible follow-up (not built): between-version
dominance claims (does v2 dominate v1: no bigger AND better) — the frontier share covers it for now.

---

## 2026-08-26 — P4-full: per-model strata + headroom selection drives the loop

**Change.** `difficulty.py`: `Difficulty.model` (None = pooled); `screen(by_model=True)` keeps each
reference model's runs apart (reads the canonical `model` from result.json, falls back to the path
slug) — pooling is only honest when one model ran, since a task hard for haiku and trivial for sonnet
would read as merely "flaky". `select_headroom(diffs, tasks, split="test", models=cfg.models)` →
`HeadroomSelection(selected, solved, unscreened)`: a task is kept when it has headroom for ANY
configured model (helping one model moves the pooled score); a task never screened for that split is
excluded, not guessed at. Judged on the **test** split — the held-out score is what the loop reports;
train headroom can't show movement there.

`loop.run_iteration(headroom_only=True, on_select=…)` narrows the task set BEFORE any agent runs
(refuses with a clear `LoopError` if nothing can move, spending nothing); `LoopResult.selection`
records the decision. `acumen loop --headroom`; `acumen screen --by-model` (model column).

**Result.** +6 tests (164): by-model strata, selection semantics (any-model, split, unscreened),
loop narrows to `hard` only with the callback firing at draft-count 0, refusal path spends nothing.
Real `screen --by-model` on the squidpy workspace: `closeness_most_central[test]@haiku` is the one
headroom task — matches the earlier diagnosis. Next: task mining (the corpus that makes P5 real).

---

## 2026-08-26 — Task mining (module 5): GitHub + web → candidates → tasks, proven live

**Change.** `mining.py` + `acumen mine` + `acumen tasks --candidates DIR`:
- **GitHub leg**: `gh search code` over the import + analysis namespaces (`sq.gr.`, `sq.im.`, …) for
  `.ipynb` and `.py`, paced at 6.5 s (the endpoint allows 10/min); each hit fetched at the commit
  the search returned (`gh api …/contents?ref=<sha>`) so origins are pinned and reproducible.
- **Web leg** (`--url`): code lifted from `<pre>` blocks / markdown fences / raw `.py`/`.ipynb`. No
  keyless web *search* API is worth depending on, so this takes URLs the operator knows rather than
  pretending to discover them — said plainly in the docstring.
- **The gate is structural, not a prompt**: keep only files that parse, reference ≥1 symbol of the
  package's *real* public API (the same inventory `acumen coverage` uses), and make ≥1 package call
  at **module level** — a library that wraps the package calls it only inside `def`s and is not an
  analysis. Excluded outright: the target repo and its `.gitmodules` submodules (the tutorials are
  already sharded — `scverse/squidpy-tutorials` slipped through the first run and prompted this),
  tests/vendored dirs, `_private` files; exact-duplicate content collapses. Every rejection carries
  a reason and `index.json` records provenance.
- Notebooks flatten to code cells (`# %%` markers; magics commented, not dropped).
- Task-gen seam: a candidate is **seeded into the shard agent's work dir** (not the source copy —
  it isn't package source) and prompted with `TASKGEN_SCOPE_MINED`: same rules as a notebook shard,
  plus "the script's own data is NOT available — re-ground every task on a bundled dataset, skip
  what no bundled dataset can support". Same shard cache, same merge, same id namespacing.

**Result.** +11 tests (175), ruff clean. **Real harvest**: one query (`sq.gr.nhood_enrichment`),
46 s → 18 candidates (13 notebooks from real labs/workshops/papers — NBIS, ELIXIR, paper-figure
repos — + 5 `.py`), 12 rejected with reasons (6 library wrappers: the top-level-call rule earning
its keep). **Live mined→task proof** (sonnet, $1.31, ~6 min): the agent read a Visium lymph-node
nhood notebook, searched squidpy's datasets module for a fit, re-grounded on `visium_hne` (bundled,
expert-annotated regions — which also sidestepped the notebook's Leiden dependency missing from the
venv), and wrote 1 task with train/test variants and analyst-defensible answers (most-enriched
neighbor of hippocampus = Pyramidal_layer; of fiber tract = Lateral_ventricle). It also persisted
its confirmation script under the namespaced id — the **first live proof of P2b** — and
`acumen coverage` now reads it: API coverage of squidpy 1.8.4.dev19+g52856c2de: 4/99 (4%)
  ground-truth scripts read from scripts (2 script(s), 1 task(s))

---

## 2026-08-27 — P5: cross-validation over tasks + the lockbox (the scientific core)

**Corpus first.** Full `acumen mine` (6 default queries × ipynb/py, paced): 692 hits → **281 candidates**
(263 new) / 411 rejected with reasons (146 library wrappers, 114 no real API symbol, 44 `_private`,
37 unparseable, 37 duplicates, 21 target/submodule, 10 tests dirs, 1 too large). Launched the task-gen
fleet on the **top 80 by API-symbol count** (13..3 symbols; 48 notebooks, 32 scripts) in the
background — ~4 h at concurrency 2, resumable by shard file; first shard landed in 7 min ($0.66).

**Why P5 exists.** The loop's train/test are *variants of the same task* — a within-task signal that
says whether a skill memorised an answer, not whether the rulebook generalises to analyses it never
saw; and every selection the loop makes on any score makes that score optimistic (selection leakage).

**Change.**
- `folds.py` (pure): `make_folds(ids, k, seed)` — ids sorted then seeded-shuffled and dealt
  round-robin, so folds depend only on the id set + seed (file order can't move a task across a
  boundary). `split_lockbox`, `write_lockbox` (**write-once**, manifest with sha256), `read_lockbox`
  (verifies the digest — an edited lockbox is an error), `check_disjoint`.
- **Guard generalised** (`improve.find_test_access`/`make_test_guard`): still denies every test split;
  now also denies a held-out task's runs in **every** split (train runs name the task and its answer)
  and any denied directory wholesale (the lockbox, all CV trees). `tests/test_guard.py` proves each
  denial through structured paths AND shell tokens — the old test-split guard had no direct test.
- `loop.run_cv_iteration`: shared parent draft+bench on the working tasks; per fold, the rulebook is
  improved from the **optimize** tasks' evidence only (structural: `collect_train_runs` is given only
  those tasks) behind the fold guard, a skill is drafted from it and scored on the **held-out** tasks;
  the CV estimate is the mean held-out Δ over folds (+ spread). The version **carried forward** is
  the **refit** on all working tasks (estimate out of sample, then fit on everything). Fold artifacts
  live in their own `cv/vN/fold-i/` roots so the linear `vN` chains and the main run tree are
  untouched; everything resumes by file presence. A **lockbox is required** unless explicitly waived
  (`--no-lockbox`, and the report then says no generalisation claim is possible); the working set
  is checked disjoint from it before any agent runs.
- `acumen lockbox --tasks --fraction --seed` (writes `lockbox/` + `<tasks>.working.yaml`);
  `acumen loop --cv K [--seed] [--lockbox DIR | --no-lockbox]` with a per-fold table, CV mean/spread,
  and the within-task number labelled optimistic.

**Result.** +13 tests (188). The CV loop tests assert the *boundary*: each fold's improve call got
exactly the optimize task ids as evidence, the held-out ids in its guard, and the CV root + lockbox
in its deny list; the refit got all tasks with nothing held out; fold scores cover exactly the
held-out tasks; the linear chain holds only v1→v2. Refusals (no lockbox / overlap) spend nothing.
Not yet run live — it needs the corpus (and P7 to score the lockbox). Next: P7-full.

---

## 2026-08-27 — P7-full: the multi-iteration loop, stopping rules, and the lockbox verdict

**Change.** `loop.run_loop` iterates `run_cv_iteration` with each iteration's parent/carried versions
**pinned** (`v{i}` → `v{i+1}`; `_ensure_rulebook(expect_version=…)` and `run_cv_iteration(parent_version=…)`),
so a resumed loop replays the same chain from disk without re-running agents — the previous
"latest version is the improved one" logic would have mis-resumed a three-version chain.
`StopRule`: `patience` (consecutive iterations whose cross-validated held-out rate does not beat the
best by `min_delta`), `max_iterations`, and a wall-clock cap checked between iterations. The pick is
the version with the best **absolute** CV held-out rate (`_cv_rates`: mean over folds of the
carried skill's held-out rate; the seed's own CV rate is the bar the first improvement must clear).
Then, once, the pick and the seed are benched on the **lockbox** tasks into `runs/lockbox/` (a tree
denied to every improve agent, so a later iteration can't learn from an earlier lockbox result);
"scored once" is file-presence resume. `LoopRun.lockbox_delta` is the one honest number.
`acumen loop --cv K --iterations N --patience P [--min-delta] [--max-hours H]` prints per-iteration
CV tables, the pick, and the LOCKBOX line. Headroom selection is made once (iteration 1) and the
same tasks are scored thereafter, or CV numbers wouldn't compare across versions.

**Result.** +2 tests (190): patience stop after a plateau, pick = v2, iteration 2's fold trees under
`cv/v3/` (parent pinned), lockbox benched exactly {v1, v2} × {z1, z2} × test in its own tree, resume
spawns zero agents and reproduces the pick; wall-clock cap stops between iterations with no lockbox
eval when waived. Ruff clean. The build plan's seven modules are now all built; the fleet is still
generating the corpus (2/80 shards at +10 min). Not yet run live: `--cv` end to end (needs the
corpus + a lockbox).

---

## 2026-08-27 — The corpus lands: 166 tasks, coverage 4 → 26 of 99

**Fleet result** (top-80 mined candidates, sonnet, concurrency 2, 01:19→08:19): **162 tasks from
74/80 shards, 162 persisted ground-truth scripts, $116.71** of subscription usage. Yield ~2.2 tasks per
candidate (many 2–3). 6 failed: 3 hit the subscription *session limit* ("resets 9:30am") — a pure
retry, relaunched with 74 shards cached; 3 agents wrote no `tasks.yaml` (nothing re-groundable on a
bundled dataset, or out of turns) — recorded, re-runnable.

**Merged corpus**: `tasks_gen.yaml` (3) + `tasks_mined.yaml` (1) + `tasks_mined_top.yaml` (162) →
`tasks_all.yaml`, **166 unique ids** (validated through the strict loader; shard namespacing made
the merge trivial). `acumen coverage --tasks tasks_all.yaml`: **26/99 (26%)** of squidpy's public
API exercised, from 162 scripts; 5 tasks have no script (3 pre-P2b + 2 where the agent didn't save
one — the soft-failure path, visible not fatal); **73 uncovered symbols** = the generation queue.

**Bench model switched to sonnet** in the workspace config (user decision): haiku never loaded
skills, which neutralised the mechanism; headroom must now come from genuinely hard tasks.

Next: retry lands → `acumen lockbox` (write-once, so only on the final set) → `bench --no-skill
--split test` on the working set (headroom only needs the test split; ~166 sonnet runs) →
`loop --cv 3 --iterations 3 --patience 2 --headroom`.

---

## 2026-08-27 — Task-gen agents stall by backgrounding (fix: the sync guard, now on task-gen too)

**Finding.** The 3 fleet shards that "wrote no tasks.yaml" all ended the same way: the agent
backgrounded a slow pipeline (Moran's I, feature extraction, a seed-robustness check), then said "I'll
wait for the notification" and ended its turn — the exact stall the bench runner already guards
against (`runner.make_sync_guard`, diary 2026-08-19), but the guard had only been wired into the
benchmark sandbox. **Fix:** `_run_generation_agent` now runs `make_sync_guard()` alongside the
skill-bias guard (both PreToolUse), and `TASKGEN_PROMPT` states the synchronous rule for all three
generators (whole / per-notebook / mined). +1 test (191). The retry fleet currently running
predates the fix; any shard it still fails will be re-run once more with it.

---

## 2026-08-27 — Final corpus (181 tasks), the lockbox is written, baseline bench launched

**Corpus complete.** Two retries finished the fleet: **80/80 shards, 177 tasks** from the top-80
mined candidates (the two shards that had stalled twice succeeded on the first run *with* the sync
guard — a direct confirmation of that fix). Merged with the 4 earlier tasks → `tasks_all.yaml`,
**181 unique ids**; coverage 26/99.

**Lockbox written (once):** `acumen lockbox --fraction 0.2 --seed 0` → **36 tasks held back**
(`lockbox/`, digest `sha256:bdf3d53d…`), **145 working** (`tasks_all.working.yaml`). From here on
nothing in the loop may read those 36; the guard denies the directory and the working set is checked
disjoint before any agent runs.

**Phase 2 running** (`run_phase2.sh`, background): warm the shared dataset cache from the 177
persisted scripts, then `bench --no-skill --split test` on the 145 working tasks with **sonnet** (the
new reference model) — the headroom baseline `loop --headroom` will select on — then
`screen --by-model`. Expect the subscription session limit to pause it once; bench resumes by file.

---

## 2026-08-27 — Baseline on the corpus: sonnet no-skill 117/145; 28 hard tasks; first live CV loop launched

**Baseline bench** (`bench --no-skill --split test`, 145 working tasks, sonnet, concurrency 2):
**117/145 passed (81%) in 2 h 17 m** ($40.19 subscription usage); 28 `wrong_answer`. `screen
--by-model` → **28 tasks with headroom**, spread over ~20 distinct source analyses (max 2 per
candidate) — the held-out pool is not one notebook's quirks. 13 datasets were pre-warmed once; the
bench's own auto-warm was then instant (cache hit) — P3 working at scale.

**Launched** (background, caffeinated, `loop.log`, `logs_loop/`):
`acumen loop --cv 3 --iterations 3 --patience 2 --max-hours 14 --headroom --lockbox lockbox`
over the 145 working tasks → headroom narrows to the 28 → 3 folds of ~9 held-out each; fresh
`skills_all/` + `rulebooks_all/` chains; `runs_all/` (holds the noskill baseline the selection reads).
Per iteration: parent draft + bench (28 × 2 splits), 3 × (fold improve → draft → held-out bench), refit
improve → draft → bench; then the one-shot lockbox eval (36 tasks × {v1, pick}). Expect ~3–4 h per
iteration; the session limit may pause it (everything resumes by file). The number to read at the
end is the **LOCKBOX Δ** line.

---

## 2026-08-27 — Session-limit hits must never become recorded runs (first live loop, paused)

**What happened.** The first live `loop --cv` drafted skill v1 and started its 56-run bench just as
the subscription **session limit** hit. `bench` records an agent error as a *completed failed run*,
so in a few minutes it wrote **54 error result.json files**; the rulebook-improve agent then failed
on the same limit and the loop exited. On resume those 54 would have been kept (file presence) and
handed to the improve agent as "the skill failed everything" — evidence about the platform, not the
skill. Purged them by hand (54 dirs; the 2 genuine runs kept) and scheduled the relaunch for after
the reset (20:42).

**Fix (structural).** `runner.is_transient` recognises session/usage/rate-limit and overload errors;
`run_once` returns such an outcome flagged `transient` **without writing result.json** (so resume
re-runs it). `bench.run_matrix` **pauses** after the first transient outcome — every still-queued run
is skipped, nothing thrown at the wall — and then raises `TransientLimitError` so no caller scores a
partial matrix; the CLI prints "paused: … rerun to resume" (exit 3). `improve_rulebook` raises the
same for a refused improve agent. +2 tests (193): order-independent (as_completed launches in
arbitrary order — my first version assumed it didn't), asserting the genuine run is the only
result.json and the queued runs never launch.

---

## 2026-08-28 — First live CV iteration: fold results, and the pause path proven under fire

**Iteration 1 folds** (28 headroom tasks, sonnet, k=3; parent = rulebook v1 / skill v1):
- fold 1: held-out **1/10 → 1/10** (+0%)
- fold 2: held-out **2/9 → 3/9** (+11%) — the first movement on analyses the improve agent never saw
- fold 3: held-out **4/9 → 4/9** (+0%)
CV estimate **+3.7%**, spread 11% — one task per fold is ~11%, so iteration 1 is within noise. Side
signal: skill v1 alone passes **7/28** of tasks the no-skill baseline failed outright (the headroom
set is 0/28 by construction) — the skill-vs-no-skill effect is real; the open question is whether
*improving the rulebook* adds to it. Refit (rulebook v2) written; skill v2 drafted.

**The pause path, live.** The session limit hit again mid-way through skill v2's 28-run bench. This
time: **0 error result.json written**, 17 of 28 runs left unrecorded, the matrix stopped launching,
`paused: … rerun the same command to resume` (exit 3). Yesterday the same event produced 54 bogus
failures. Relaunch scheduled for after the reset (01:42); iteration 1 completes with the remaining
17 runs, then iteration 2 (parent v2 → v3) begins.

**Also today:** `coverage --skill` (taught vs verified, bare-name credit): skill v1 teaches 53/99
symbols, the benchmark vouches for 24 — 29 taught-but-unverified is the phase-3 target list.
`run_phase3.sh` prepared (tutorial notebooks → working set; lockbox untouched).

---

## 2026-08-28 — Iteration 1 complete; a latent bug the multi-iteration loop exposed (diff inside the version dir)

**Iteration 1 (rulebook v1 → v2), cross-validated over 28 headroom tasks:** CV **Δ pass +3.7%**
(spread 11.1%), **Δ load +10.7%**; within-task test (optimistic): noskill 0/28 → skill v1 7/28 →
skill v2 6/28. Rationale of v2: added "test sensitivity, not just existence" guidance to the drafting
rules. Honest read: no measurable pass gain yet; the load-rate gain is the earlier signal moving.

**Bug.** Starting iteration 2 the loop refused to load v2: *"modified since it was written"*. The
CLI wrote `from-v1.diff` **inside** `rulebooks/v2/` after `meta.json` had hashed the directory — the
hash covers every content file, so the version read as tampered. Invisible in single-iteration runs
(v2 was never reloaded); the tamper check did exactly its job. **Fix:** `rulebooks.diff_path` puts
diffs under `<root>/diffs/<version>-from-<parent>.diff`, both CLI sites use it, +1 regression test
(197). Moved the stray file in the workspace; v2 verifies again (`sha256:0172869f…`). Relaunched.

---

## 2026-08-28 — Iteration 2 (v2 → v3); the pause path holds a second time

**Iteration 2 folds** (parent = rulebook v2 / skill v2): fold 1 **1/10 → 1/10**, fold 2 **2/9 → 2/9**,
fold 3 **3/9 → 5/9 (+22%)**. CV **Δ pass +7.4%** (spread 22%), Δ load −7%. Within-task test:
v2 6/28 → v3 9/28. Rationale (v3): "12/28 train failures were all wrong_answer with the skill loaded
(86% load) — the body, not the description, was the lever; three recurring, fixable mistakes in the
failing scripts…" — evidence-driven and legible.

**Selection so far** (absolute CV held-out rate, mean over folds of the carried skill): v1 ≈ 26%,
v2 ≈ 29%, v3 ≈ 29% (tie, not > best) → pick stays **v2**, one non-improving iteration against
patience 2. Iteration 3 (v3 → v4) is the last allowed; then the lockbox (36 tasks × {v1, pick}).

**Observation.** Every gain so far is one fold with 1–2 extra tasks passing; spreads (11–22%) dwarf
the means. Nine-or-ten held-out tasks per fold cannot resolve a few-percent effect — the corpus
needs to grow (phase 3) before the CV number is worth quoting, and the lockbox (36) is the better
instrument this run has. Session limit paused the loop a third time (resets 06:40); again 0 bogus
results, 19/28 runs unrecorded, relaunch scheduled.

---

## 2026-08-28 — Iteration 3 (v3 → v4) is negative; the loop stops on patience; lockbox pending

**Iteration 3 folds** (parent v3): fold 1 **3/10 → 2/10**, fold 2 **2/9 → 0/9**, fold 3 **4/9 → 3/9**
— all negative, CV **Δ pass −14.4%** (spread 12%), Δ load −3.7%; within-task v3 9/28 → v4 6/28.
Improving on v3's train evidence made the drafted skill *worse* on analyses it hadn't seen: the first
clear out-of-sample signal of the run, and the kind of thing within-task scoring could not have shown
(v3's within-task 9/28 looked like the best version).

**Stop + pick.** Two non-improving iterations (v3 tie, v4 worse) → patience fires; pick by CV =
**v2**. Lockbox evaluation started (36 tasks × {v1, v2}); the session limit paused it at 9/36 v1 runs
(0 bogus results, fourth clean pause); relaunch 11:42 → LOCKBOX Δ expected ~13:30.

**Rhythm learned.** Sonnet at concurrency 2 exhausts the subscription's session window roughly every
5 h of wall-clock; the loop loses ~2 h per pause (reset wait). The pause/resume path has now
carried the run across four pauses with no lost or corrupted work.

---

## 2026-08-28 — The first live CV loop finishes: pick v2, LOCKBOX +3%

**Result** (`loop --cv 3 --iterations 3 --patience 2 --headroom`, 28 hard tasks, sonnet bench; ran
19:45 → 13:07 across four session-limit pauses):

| iter | rulebook | CV Δ pass | spread | CV Δ load | within-task (optimistic) | size |
|---|---|---|---|---|---|---|
| 1 | v1 → v2 | +3.7% | 11% | +10.7% | v1 7/28 → v2 6/28 | 31.1 → 25.6 KB |
| 2 | v2 → v3 | +7.4% | 22% | −7.0% | v2 6/28 → v3 9/28 | → 27.6 KB |
| 3 | v3 → v4 | **−14.4%** | 12% | −3.7% | v3 9/28 → v4 6/28 | → 29.7 KB |

Absolute CV held-out rate: v1 ≈ 26%, v2 ≈ 29%, v3 ≈ 29% (tie), v4 ≈ 15% → **pick v2**; stopped on
patience. **LOCKBOX (36 tasks never read by anything in the loop): v1 26/36 (72%) → v2 27/36 (75%),
Δ +3%** — two tasks gained (mibitof co-localization, nuclei-max-spot), one lost (H&E Moran top gene).
v2 is also the *smallest* version (25.6 KB vs v1's 31.1 KB): leaner and no worse.

**Honest reading.** (1) The mechanism works end to end on real data: structural hold-out, pinned
resume across four pauses, a pick made only on CV, a lockbox opened once. (2) The effect is within
noise: one task on the lockbox; one or two tasks per fold in CV. Nine-or-ten held-out tasks per fold
cannot resolve a few-percent effect, and the lockbox is ~72% baseline-solvable so most of its 36
carry no information about the skill. (3) The one *clear* signal was negative: v4 regressed −14%
out of sample while looking best within-task (v3 9/28) — exactly the failure mode CV exists to
catch, and evidence the within-task number would have misled. (4) The improve agent's edits are
legible and evidence-driven (v3's rationale diagnosed body-vs-description from the 86% load rate).

**What decides the next run:** more headroom tasks per fold (phase 3: tutorial notebooks → the 29
taught-but-unverified symbols), a noskill bench on the lockbox for the floor, and reporting the
lockbox Δ on its hard subset. Cost of this run: ~$210 subscription usage over four session windows.

---

## 2026-08-29 — Lockbox floor; phase 3 (tutorial notebooks) launched

**Floor.** `bench --no-skill` on the 36 lockbox tasks (test split, sonnet): **22/36 (61%)**. The full
lockbox line is now **noskill 22/36 → skill v1 26/36 → rulebook-improved v2 27/36**: the skill itself
is worth +4 tasks (+11%) on analyses nothing in the loop ever read; one more from the rulebook
iteration. Same caveat as before — one task is 3%.

**Phase 3 launched** (`run_phase3.sh`, after the floor so the two never share the session window):
the 50 bundled tutorial notebooks through sharded task-gen (first time at scale — P1's proof was one
notebook), merged into the *working* set only (the lockbox is write-once and stays exactly as it is),
then coverage before/after. Target: skill v1's 29 taught-but-unverified symbols (niche, sepal, image
container, experimental stain/tiling QC). One `chmod +x` I had forgotten cost a 30-minute gap.

---

## 2026-08-29 — The stall's real cause: the CLI backgrounds long Bash calls on timeout

A tutorial shard (CellProfiler) stalled again *with* the sync guard active and **zero denied tool
calls**: "the full-image computation is running in the background (~10 min)… I'll pause here until
it completes." The agent never asked to background anything — the Claude Code harness moves a Bash
command that outlives its timeout (2 min default) to the background on its own and tells the agent
to await a notification that a one-shot query never delivers. No PreToolUse input carries a flag to
deny, so the guard cannot see it. **Fix upstream:** every isolated agent env now sets
`BASH_DEFAULT_TIMEOUT_MS` / `BASH_MAX_TIMEOUT_MS` to 45 min (`env.BASH_TIMEOUT_MS`), so a real
analysis step (segmentation, permutation tests) runs inline. Both arms identical (parity). +1 test.
The running phase-3 fleet imported the old module; shards it fails are re-run afterwards with this.

---

## 2026-08-29 — Phase 3 run 1: 33/50 notebooks → 51 tasks, coverage 26 → 29/99; task-gen learns to pause

**Run 1** (50 tutorial notebooks, sonnet, concurrency 2): **51 tasks from 33 shards**; 17 failed —
16 on the subscription session limit (resets 07:20), one `ProcessError` (`tutorial_tf`), one
CellProfiler stall (the harness-timeout backgrounding, fixed above but not yet live in that fleet).
Merged into `tasks_all_v2.working.yaml` (145 working + 51 = 196; lockbox untouched). Coverage
**26 → 29/99** from a third of the notebooks; relaunch scheduled for 07:23 (the 17 missing shards
resume; cached ones skip).

**Change.** The sharded generator had no pause logic: after the limit hit it failed the remaining
shards one by one against the same wall (harmless — nothing written — but wasteful, and noisy).
`generate_tasks_sharded` now pauses after the first transient shard failure (the rest are
`skipped:`), still merges what landed, and reports `paused`; the CLI prints the summary then raises
`TransientLimitError` (exit 3). +1 test (199), order-independent (max_concurrency=1).

---

## 2026-08-29 — Phase 3 complete: 50/50 notebooks → 83 tasks, working set 228; round 2 launched

**Phase 3 run 2** (07:22–09:07, the 17 shards run 1 had left; pause logic + 45-min Bash timeout
live): all 17 landed, including the CellProfiler notebook that had stalled twice and `tutorial_tf`.
**50/50 notebooks → 83 tasks.** Merged into the working set only: **228 working tasks**, lockbox
still exactly its 36. Coverage **29 → 30/99** — smaller than hoped: the notebooks' ground-truth
scripts exercise mostly the same core calls the mined corpus already did; the remaining queue is
dominated by `experimental.*` (no public usage exists), `gr.neighbors.*` builders, and `pl.*`.

**Round 2 launched** (`run_round2.sh`): (1) baseline `bench --no-skill --split test` for the 83 new
tasks (the 145 old ones resume from cache); (2) **fresh roots** — `rulebooks_r2/v1` seeded from
round 1's CV pick (`rulebooks_all/v2`, provenance recorded), `skills_r2/`, `runs_r2/` with the
no-skill baseline copied in — so round 1's version chain and fold trees are never mixed with a
different task set; (3) `loop --cv 3 --iterations 3 --patience 2 --headroom` over the 228, same
lockbox. Each agent step retries every 30 min on a session-limit pause. The point of the round:
more headroom tasks per fold, so a few-percent effect can actually be resolved.

---

## 2026-08-30 — Round 2 on the 228-task corpus: iterations 1–2

**Setup.** Baseline on the 83 notebook tasks: 66 ok / 15 wrong / 2 error → **45 headroom tasks**
(15 per fold, up from 9–10). Round-2 `v1` = round 1's pick (`rulebooks_all/v2`), fresh roots.

| iter | rulebook | folds (held-out Δ) | CV Δ pass | within-task |
|---|---|---|---|---|
| 1 | r2-v1 → r2-v2 | +0%, −7%, **+20%** | **+4.4%** (spread 27%) | v1 13/45 → v2 18/45 |
| 2 | r2-v2 → r2-v3 | **−27%**, +7%, **−20%** | **≈ −13%** | pending (bench paused at 22/45) |

Same shape as round 1: one improvement step from the seed helps a little (carried by one fold), the
next step from that improved version *hurts* out of sample — and by a lot more than it helped. The
improve agent, given a train set where the skill already passes ~40%, over-fits the remaining
failures into rules that break analyses it never saw. Iteration 2's rationale again edited the body.

**Ops.** Four session-limit pauses in this round so far; the `retry` wrapper (12 × 30 min) covered
three, but a 03:50 reset falls past its last attempt, so a manual relaunch is scheduled for 03:52.
Consider making the wrapper parse the reset time instead of polling blindly.

---

## 2026-08-30 — Round 2 iteration 2 confirmed (−13%); a colon in a description killed a fold draft

**Iteration 2 (r2-v2 → r2-v3):** CV **−13.3%** (spread 33%), Δ load +11%; within-task v2 18/45 →
v3 12/45. Iteration 3, fold 1: **3/15 → 6/15 (+20%)** — then fold 2's *draft* failed: the agent wrote
a `description:` with an unquoted colon, YAML rejected the frontmatter, `DraftError` (exit 2) ended
the run. The content was fine; only quoting was missing — and the frontmatter is *our* format
constraint. **Fix:** `skills.normalize_frontmatter` re-quotes unquoted top-level scalars when (and
only when) the frontmatter fails to parse; `draft._validate_staged` applies it to the staged
SKILL.md before validation. +1 test (200). Relaunched; the loop resumes at that draft.

**Ops note.** `pgrep -f run_round2.sh` inside a monitor matches the monitor's own command line, so
those monitors never see the exit; key liveness on the real `acumen loop --config` process.

---

## 2026-08-30 — Round 2 verdict: pick r2-v4, lockbox 22 → 29/36 — and draft variance is the real finding

**Round 2** (45 headroom tasks, k=3, seed = round 1's pick): iter 1 CV +4.4%, iter 2 **−13.3%**,
iter 3 **+11.1%** (all three folds positive) → carried r2-v4 at 38% CV beats r2-v2's 33% → **pick r2-v4**
(max iterations). **LOCKBOX: r2-v1 22/36 (61%) → r2-v4 29/36 (81%), +8 gained / −1 lost.**

**But:** r2-v1 is a *fresh draft* of the same rulebook text as round 1's v2, which scored 27/36. Same
text, two drafts, **9 of 36 lockbox tasks differ** (27 vs 22). Draft-to-draft variance is as large as
the effects the loop measures. Honest statements: r2-v4 vs no skill **22 → 29 (+8/−1)**; r2-v4 vs
round 1's best skill **27 → 29 (+3/−1)** — the best artifact so far, with the CV pick and the lockbox
agreeing, but the "+19%" headline includes a draft that happened to land low. r2-v4 is 33.3 KB vs
the 24.2 KB seed: this round bought passes with size.

**What this changes.** The rulebook is scored through *one* draft. That makes every CV and lockbox
number a sample of size one from a distribution whose spread we now know is ~±5/36. Next design
lever: score a rulebook by the **mean over N drafts** (draft variance becomes a reported quantity, and
the pick is by the mean), and read the size axis per draft. Second lever, unchanged: the improve
step's second iteration regressed in both rounds — over-fitting residual failures — so evidence
selection for the improve agent deserves work too.

Ops: ~6 session-limit pauses across the round; all clean; the retry wrapper + one manual relaunch.
Cost: roughly $300 subscription usage for round 2.

## 2026-08-31 — N-draft rulebook scoring (task A of the playbook)

Draft variance (9/36 lockbox tasks between two drafts of identical rulebook text) made every
single-draft number a sample of one. Now the lockbox verdict can be scored over N independent
drafts per version: `loop --cv K --drafts N`.

Design (as staged in tasks/next-session.md): draft 1 stays the primary `skills/vK` (lockstep
untouched); drafts 2..N live at `skills/drafts/vK/d<i>/` — each its own skills-root holding a
single v1 — and bench into `runs/drafts/vK/d<i>/lockbox/`, so no arm collides and the primary's
existing lockbox runs keep their paths (a resumed run with `--drafts 3` only adds the two missing
drafts). `runs/drafts` joined the CV/lockbox trees in every improve agent's deny_dirs.
`DraftScores` reports per-draft rates + sizes, mean, and spread; `LoopRun` keeps the primary
`lockbox_score`/`lockbox_baseline` fields and adds the draft sets plus `lockbox_mean_delta` (== the
old delta at N=1). Staged scope, documented: CV folds — and therefore the pick — stay single-draft;
extend only if the variance experiment shows fold noise is dominated by draft noise.

Tests: 204 (was 201) — mean/spread math, n_drafts validation, and a full-loop test asserting the
variant layout, per-draft lockbox benches in their own trees, deny_dirs, primary-path stability,
and agent-free resume. Not yet run live: the variance experiment (B) is the first consumer — with
this landed it's just a rerun of the finished round-2 loop with `--drafts 3` appended (resume
leaves v1/v4 primaries cached; ~4 h of lockbox benches for the 4 new drafts).

## 2026-08-31 — `acumen evolve`: generational improve-from-best (task C, reshaped)

The user's steer: hundreds of generations of potentially wild rulebook edits, reliably measured —
the CV loop's ~6 h/iteration can't carry that, and its chain walks through regressions. New module
`evolve.py`:

- **Improve from the best, always** — every generation's improve agent starts from the current
  champion; rejected candidates keep their version dirs but nothing descends from them.
- **Rotating exploration directives** (8: focused repair, bold restructure, aggressive deletion,
  concretization, loading/routing, assumption-inversion, generalization, free choice) injected as
  maintainer feedback — recorded in each version's meta.json.
- **Two-tier selection**: cheap screen (12-task rotating subset, seeded per epoch) decides
  accept/reject at `accept_delta` passes (noise floor — calibrate from the variance experiment);
  the *confirmed* champion only changes on a full headroom-test bench every `confirm_every`
  accepts, and a failed confirmation reverts. Screens may thrash; the ratchet may not.
- **Archive**: one JSON line per generation (`runs/evolve.jsonl`) — directive, versions, subset,
  scores, decision — the dataset a cross-pollination step will mine.
- **Determinism = resume**: directive, subset, and candidate version are pure functions of
  (generation, seed); a resumed run replays every decision from on-disk scores, zero agents.
- Lockbox unchanged: opened once at the end, champion + seed over `--drafts` (default 3);
  `evaluate_lockbox=False` for future island runs, which never open it.

Cost/generation ≈ improve + draft + ~12–24 screen benches ≈ 45–60 min at 2-concurrency.
Next layer (designed with the user, not yet built): **islands** — k independent evolve runs on
disjoint task partitions, then a cross-pollination meta-agent that distills edits replicating
across ≥2 islands into meta-rules; validation = an edit that won on island A must hold on island
B's tasks. Only the merged champion opens the lockbox.

210 tests (was 204): directive/subset determinism, improve-from-best after reject AND after a
reverted confirmation, ratchet promote/revert, journal idempotence, agent-free resume, parameter
validation, lockbox denial. Not yet run live (variance experiment occupies the session window).
