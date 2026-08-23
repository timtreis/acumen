# acumen — rulebook autoresearch loop TODO

Goal: a continuous loop that optimizes a versioned **rulebook** (the instructions that generate
`SKILL.md`) for **squidpy**, scored on held-out tasks under CV, stratified by difficulty, over tasks
that exhaustively cover the package. Rulebook = the optimized artifact; skills = intermediates.
Binding constraints: wall-clock and selection leakage (NOT dollar cost). Full design: `diary.md`
(2026-08-15 target-design entry).

Build order: P0 submodules → P1 sharded task-gen → P2 coverage → P3 warm cache → P4 difficulty →
P5 CV+lockbox → P6 rulebook artifact → P7 loop.

## Done (build order)

- [x] **P0 — clone target submodules.** `env.py:_clone` recurses submodules after ref checkout;
      squidpy's 50 notebooks now check out. Marker records `submodules`. (`env.py`, `config.py`)
- [x] **P1 — sharded task generation.** One agent per notebook, fanned out under a semaphore, each
      writing a validated `shards_dir/<slug>.yaml`; per-shard failure isolation, resume by file
      presence, `merge_shards` namespaces ids. Prompt split via `{scope}`/`{coverage_check}` seam.
      `acumen tasks --per-notebook [--shards-dir] [--notebook SUBSTR]`. (`taskgen.py`, `prompts.py`,
      `cli.py`, `tests/test_taskgen.py`)
- [x] **P1 proven on real squidpy.** First real task-gen agent run on `compute_centrality_scores`:
      3 well-formed tasks, **6/6 ground-truth answers independently verified correct**.
- [x] **P6 (crude) — rulebook artifact.** `rulebooks/vN/rulebook.md` = draft-prompt template text;
      v1 seeded from `DRAFT_PROMPT`; `validate_rulebook` guards placeholders. `draft_skill(rulebook=)`
      + `draft_prompt(template=)` seam. (`rulebooks.py`, `prompts.py`, `draft.py`)
- [x] **P7 (crude) — the loop.** `run_iteration`: seed v1 → draft → bench → improve rulebook → draft
      v2 → bench → report. `improve_rulebook` mirrors `improve_skill` one level up. `acumen loop`.
      Resumable by file presence. (`loop.py`, `prompts.py:RULEBOOK_IMPROVE_PROMPT`, `cli.py`,
      `tests/test_loop.py`)
- [x] **P4 (partial) — difficulty screening.** `difficulty.py:screen()` reads baseline (noskill) arm
      pass rate → strata (solved/flaky/hard); `has_headroom`. `acumen screen`. (`tests/test_difficulty.py`)
- [x] **Reliable bench.** PreToolUse sync-guard denies `run_in_background`/monitor tools in the
      sandbox (both arms, parity-safe) + preamble line. Stops the background-stall failure.
      (`runner.py:find_background_use/make_sync_guard`, `prompts.py` HARNESS_PREAMBLE)
- [x] **Bench runs on a subscription.** Dropped `allow_session` from `resolve_auth_mode`; `bench`
      got `--auth`. Cost is not an optimized metric. (`env.py`, `cli.py`)
- [x] **Loop metrics = PASS + LOAD.** `Score` carries `loaded`/`load_rate`; loop reports
      noskill → skill v1 → skill v2 on both pass and load, with `moved`/`load_moved`. (`loop.py`, `cli.py`)

## First measured loop run (2026-08-19) — outcome

- [x] Ran the loop end-to-end on a headroom task (closeness centrality, haiku bench / sonnet
      authoring). Result: **noskill 0/1 → skill v1 0/1 → skill v2 0/1, no movement**, PASS and LOAD.
      Machinery + improve-agent reasoning sound; **blocker is task calibration**, not code (see below).

## Open — critical path to a *measurable* loop

- [ ] **Task-gen quality: analyst-defensible answers.** Task-gen's "run it, take the top" produced an
      adversarial answer (`Low quality`, a QC-artifact group nobody reports). Tighten the generation
      rulebook / verification so generated answers are what a competent analyst reports (reject
      QC/ambiguous groups, near-ties). This is the direct next step. (touches `prompts.py:TASKGEN_PROMPT`)
- [ ] **Calibrate difficulty against a skill-LOADING model (sonnet), not haiku.** haiku ran
      `skill_loaded=False` on every arm → skill mechanism neutralized. Weak-model shortcut fails.
      Need tasks hard *for sonnet* (sonnet solved all easy centrality tasks 6/6). Generate/curate
      harder analyses (image features, ligrec, goal-only phrasing) and `acumen screen` a batch.
- [ ] **Investigate why skills under-load** before over-investing in descriptions: is it description
      wording (fixable by the rulebook) or model behavior (haiku just doesn't load)? Check load rates
      per model on a task whose description names the capability.
- [ ] Consider hardening the bench sandbox further and/or a task-weight guard: closeness centrality
      on 19k cells took ~14 min/run. Favor cheap analyses in generation, or cap graph size.

## Open — remaining build order (real versions)

- [ ] **P2 — coverage measurement.** Instrument ground-truth `script.py` execution to record which
      squidpy API symbols were actually called → verified coverage; uncovered symbols → generation
      queue. (~43 analysis-bearing symbols; see diary.)
- [ ] **P3 — warm dataset cache.** Each bench run re-downloads its squidpy dataset (scrubbed
      HOME/XDG_CACHE_HOME). Share a warm cache without loosening the env scrub.
- [ ] **P4 (full) — difficulty strata as a pipeline.** Fix + record the reference model; strata drive
      task selection for the loop's held-out.
- [ ] **P5 — CV over tasks + selection-leakage lockbox.** Genuinely new axis (today's train/test is
      within-task variants). Needs nested CV or a structurally unreadable lockbox. Load-bearing.
- [ ] **P6 (full) — rulebook artifact like `skills.py`.** Content-hashed, immutable, `meta.json`
      provenance chain (crude version is just a versioned file).
- [ ] **P7 (full) — the loop.** Multi-iteration, stopping rules, hard caps; restructure `loop.py`
      control flow once P5 lands.
- [ ] **Large-scale task mining.** The intended task pool is mined from GitHub + the web at scale
      (real squidpy analyses), not just the 50 notebooks. Distinct, unbuilt subsystem.
- [ ] **Leanness Pareto axis = skill SIZE, not cost.** `report.py` frontier is currently
      cost-vs-success; under the no-cost steering it should become size-vs-success.

## Notes for the next session

- Workspace: `~/acumen-squidpy/` (config, tasks_gen.yaml, rulebooks_gen/, skills_gen/, runs_gen*/,
  loop.log). squidpy target cached under `~/.cache/acumen/squidpy-*`.
- Auth: acumen can't read the macOS Keychain login; run `claude setup-token`, extract the
  `sk-ant-oat01-…` token, export `CLAUDE_CODE_OAUTH_TOKEN`, run with `--auth session`. See the
  `squidpy-loop-workspace` memory.
- Wrap long runs in `caffeinate -i` — the Mac slept mid-draft once and killed a run.
