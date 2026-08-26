# acumen — rulebook autoresearch loop TODO

Goal: a continuous loop that optimizes a versioned **rulebook** (the instructions that generate
`SKILL.md`) for **squidpy**, scored on held-out tasks under CV, stratified by difficulty, over tasks
that exhaustively cover the package. Rulebook = the optimized artifact; skills = intermediates.
Binding constraints: wall-clock and selection leakage (NOT dollar cost). Full design: `diary.md`;
build plan + decisions: `tasks/build-plan.md`.

## Built (all seven modules of the build plan, 2026-08-26/27) — see diary for each

- [x] **P0** submodules · **P1** sharded task-gen (proven live) · **P2** coverage (`acumen coverage`,
      persisted ground-truth scripts; proven live) · **P3** warm dataset cache (`acumen warm`; proven
      live) · **P4** per-model difficulty strata + `loop --headroom` (proven on real runs) ·
      **P6** content-hashed/immutable/provenanced rulebook · **leanness = skill size** in the report
      (proven on real runs) · **mining** (`acumen mine`, `tasks --candidates`; proven live: 281
      candidates, mined→task→script chain) · **P5** CV folds + write-once lockbox + hold-out guard ·
      **P7** `run_loop` with stopping rules, CV pick, one-shot lockbox verdict (`loop --cv`).
- [x] Reliable bench (sync-guard), bench on subscription, PASS + LOAD loop metrics. 190 tests.

## In flight

- [ ] **Corpus fleet** (background, `~/acumen-squidpy/mined_top_taskgen.log`): task-gen over the top-80
      mined candidates → `tasks_mined_top.yaml` + `scripts/`. Resumable: re-run the same command to
      continue. Then merge with `tasks_gen.yaml` + `tasks_mined.yaml` by hand (distinct ids already).
      The remaining ~200 candidates in `mined/` can be run later the same way.

## Next (the first honest measurement)

- [ ] Merge the corpus → `tasks_all.yaml`; `acumen coverage` on it (expect a big jump from 4/99).
- [ ] `acumen lockbox --tasks tasks_all.yaml --fraction 0.2` → `lockbox/` + `tasks_all.working.yaml`.
- [ ] `acumen bench --no-skill --tasks tasks_all.working.yaml` (baseline for headroom; haiku per config).
- [ ] `acumen loop --cv 3 --iterations 3 --patience 2 --headroom --tasks tasks_all.working.yaml
      --lockbox lockbox --auth session` (caffeinate; hours). First live `--cv`; watch for: fold agents
      being *denied* held-out paths in their logs (proof the guard bites), CV mean/spread, LOCKBOX Δ.
- [ ] Reference model: haiku doesn't load skills (LOAD 0) — consider `models: [claude-sonnet-5]` for
      the bench arm before the long run, else the mechanism is neutralised (see memory).

## Later / open

- [ ] Task-gen answer defensibility (reject QC-artifact groups, near-ties) — the mined prompt helped;
      still no structural check.
- [ ] Between-version dominance in the report (does v3 dominate v2: no bigger AND better).
- [ ] Remaining mined candidates; web-leg URLs (`mine --url`) for tutorials outside GitHub.
- [ ] `tasks --max-concurrency` flag (fleet concurrency is config-only today).

## Notes

- Workspace `~/acumen-squidpy/` (config with `dataset_cache_dirs: [data, cache]`, `mined/`,
  `mined_top/`, `tasks_*.yaml`, `scripts/`, `logs_mined/`). Auth: `CLAUDE_CODE_OAUTH_TOKEN` from
  `.token`; `--auth session`; wrap long runs in `caffeinate -i`. See the `squidpy-loop-workspace` memory.
- Lint: `uvx ruff@0.16.1 check src tests` / `format --check`; never commit a reformat of
  `src/acumen/_skills/data/references/python-api.md`.
