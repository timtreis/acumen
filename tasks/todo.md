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

## Done 2026-08-27/28 — the first honest measurement

- [x] Corpus: 80/80 mined shards → 177 tasks; merged → `tasks_all.yaml` **181 tasks**, coverage 26/99.
- [x] Lockbox written once (36 tasks); working set 145. Bench model switched to sonnet.
- [x] Baseline `bench --no-skill --split test`: sonnet 117/145; **28 hard** tasks (headroom).
- [x] First live `loop --cv 3 --iterations 3 --patience 2 --headroom`: v1→v2 CV +3.7%, v2→v3 +7.4%
      (tie on absolute rate), v3→v4 **−14.4%**; pick **v2**; **LOCKBOX v1 26/36 → v2 27/36 (+3%)**.
      Four session-limit pauses, all clean (0 bogus results) after the transient-limit fix.
- [x] Bugs found live and fixed: task-gen stall (sync guard), limit-as-failure (pause + raise),
      diff-inside-version-dir (hash tamper false positive).

## Next

- [ ] **Phase 3 — grow the corpus where it is gradeable** (`~/acumen-squidpy/run_phase3.sh`): the 50
      tutorial notebooks through sharded task-gen → working set (lockbox untouched) → coverage. Then
      targeted backfill from `coverage --queue` (`mine --query`, `tasks --feedback`), remaining 201
      candidates. Goal: turn skill v1's 29 taught-but-unverified symbols into verified ones.
- [ ] Re-bench `--no-skill` on new tasks; rerun the loop from the carried rulebook (v2) with more
      headroom tasks per fold — 9–10 per fold cannot resolve a few-percent effect.
- [ ] Lockbox context: bench `noskill` on the lockbox tasks once (36 runs) so the LOCKBOX line can
      also show the skill-vs-no-skill floor; consider reporting the lockbox Δ on its hard subset.
- [ ] Never run task-gen fleets and the loop concurrently (shared session window).

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
