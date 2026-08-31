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

## Round 2 (2026-08-29/30) — done

- [x] Phase 3: 50 notebooks → 83 tasks; working set 228; coverage 30/99; skill coverage 24 → 28 verified.
- [x] Round 2 loop (fresh roots seeded from r1 pick): iter CVs +4.4% / −13.3% / +11.1%; pick **r2-v4**;
      LOCKBOX r2-v1 22/36 → r2-v4 29/36 (+8/−1); vs round-1 best 27 → 29.
- [x] **Finding: draft variance.** Same rulebook text, two drafts: 9/36 lockbox tasks differ (27 vs 22).
- [x] Fixes found live: task-gen pause, 45-min Bash timeout (harness backgrounding), frontmatter
      normalization, transient draft failures as pauses.

## Next  — detailed playbook: `tasks/next-session.md`
- [ ] **Score a rulebook over N drafts** (mean ± spread; pick by mean). Draft variance ≈ effect size
      today, so every single-draft number is a sample of one. Loop change in `_ensure_skill`/`_bench`
      + report. Then re-measure r2-v4 vs r2-v1 with N=3 on the lockbox to see what survives.
      **Awaiting the user's "build it" (proposed 2026-08-30, end of round 2); the variance experiment
      (~216 lockbox runs ≈ 4 h ≈ one session window) follows on a separate go.**
- [ ] Fix the stale `paused` retry ergonomics: `run_round2.sh`'s blind 30-min retry misses resets that
      fall past its last attempt; parse the `resets HH:MM` from the CLI message instead.

- [ ] Improve-step evidence: in both rounds the 2nd iteration regressed out of sample (over-fitting
      residual train failures). Options: improve from fold-external failures only, cap edit size,
      or require CV gain before carrying. Decide after N-draft scoring lands.
- [ ] Targeted coverage backfill: `coverage --queue` → `tasks --feedback` naming the 21 taught-but-
      unverified symbols (mostly `calculate_niche*`, loaders); remaining 201 mined candidates.
- [ ] Report the lockbox Δ on its hard subset (14 tasks noskill fails) beside the full 36.
- [x] Lockbox floor benched once: noskill 22/36.
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
