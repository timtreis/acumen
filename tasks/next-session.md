# Next session — detailed plan (written 2026-08-31, after rounds 1–2)

Read `tasks/todo.md` for the ledger and `tasks/lessons.md` before touching agents. This file is the
operational playbook for the next work session. Diary has the full history; the
`loop-findings-rounds-1-2` memory has the headline numbers.

## State snapshot

- Repo clean at `origin/claude/acumen-overview-c410ls`; 201 tests; ruff pinned 0.16.1.
- Workspace `~/acumen-squidpy/`: `tasks_all_v2.working.yaml` (**228 working tasks**), `lockbox/`
  (**36 tasks, write-once, digest-verified**, noskill floor **22/36**), `mined/` (281 candidates,
  ~200 unrun), `scripts/` (245 ground-truth scripts), `runs_all/` (round 1 + baselines),
  `runs_r2/` (round 2), `rulebooks_all/` (v1–v4, pick **v2**), `rulebooks_r2/` (v1–v4, pick
  **r2-v4**, seeded from r1-v2), `skills_r2/v4` = best skill (lockbox 29/36).
- Key numbers: lockbox noskill 22 / r1-v1 26 / r1-v2 27 / r2-v1 22 / **r2-v4 29** of 36.
  **Draft variance: same rulebook text drafted twice → 27 vs 22, 9/36 tasks differ.**
- Auth: `export CLAUDE_CODE_OAUTH_TOKEN="$(cat ~/acumen-squidpy/.token)"`, `--auth session`.
  Session limit: ~5 h of 2-concurrency sonnet per window; paused runs exit 3 and resume by rerun.

## A. Build N-draft rulebook scoring  ← the blocker for every further conclusion

Goal: a rulebook version's score = **mean ± spread over N drafted skills**, so draft variance is a
reported quantity instead of a confound. Awaiting-user status: proposed end of round 2; user said
"write the todo" — confirm "build it" before starting round 3, but the code can be built cold.

Design (decided far enough to implement; revisit against the code):
- Layout mirrors the CV fold pattern (nested roots keep the `vN` naming contract intact):
  extra drafts of version `vK` live at `skills_root/drafts/vK/d2/…/v1` (each `d<i>` its own
  skills-root holding a single `v1`); the primary draft stays `skills_root/vK` (rulebook↔skill
  lockstep preserved). Their benches go to `runs_root/drafts/vK/d<i>/` (own run trees, no arm
  collisions). Add `runs_root/drafts` to every improve agent's `deny_dirs`.
- `loop.py`: `_ensure_skill_variants(n)` → list[Skill] (resume by presence, like folds);
  score helpers return `DraftScores(per_draft: list[Score], mean_rate, spread)`; `run_loop` picks
  by mean; `_lockbox_eval` evaluates every draft of the pick (and seed) or at least N of them.
- **Scope decision (stage it):** version-level scores (parent / carried / lockbox) get N drafts
  first; CV folds stay single-draft initially — k×N would multiply agent cost ~3×. Document the
  limitation in the docstring; extend to folds only if the experiment (B) shows fold noise is
  dominated by draft noise.
- CLI: `loop --drafts N` (default 1 = today's behaviour); print per-draft rates + sizes.
- Tests to write: variant layout + resume (no agents respawned), mean/spread math, deny_dirs
  includes the drafts tree, lockbox evaluates each draft once (file presence).

## B. Variance experiment (~216 lockbox runs ≈ one session window, needs user go)

Purpose: how much of lockbox 22 → 29 survives averaging? Decides whether round 3 (~$300/~24 h) is
worth it. Can run with TODAY'S code (no A needed) via ad-hoc roots:

```zsh
cd ~/acumen-squidpy && export CLAUDE_CODE_OAUTH_TOKEN="$(cat .token)"
# 3 extra drafts each for r2-v1 and r2-v4 (python, uses draft_skill(rulebook=text) directly):
#   for ver in v1 v4: text = rb.load_rulebook("rulebooks_r2", ver).text
#     for i in 2 3 4: draft into exp_drafts/<ver>/d<i>  (skills-root per draft, version v1)
# then per draft:
#   acumen bench --config config.yaml --tasks lockbox/tasks.yaml --runs runs_exp/<ver>-d<i> \
#     --skill v1 --skills exp_drafts/<ver>/d<i> --split test --auth session
```
Analysis: per-rulebook mean ± spread over {existing draft + 3 new} on the 36 lockbox tasks; also the
hard-subset (14 tasks noskill fails) view. Report honestly even if the effect shrinks.

## C. Improve-step evidence (after A/B)

Both rounds: iteration 2 regressed out of sample (r1: −14.4%, r2: −13.3%) — the improve agent
over-fits residual train failures once the skill passes ~40%. Options to weigh (pick ONE, test on
CV): (1) show the improve agent only *failures the parent also had* (stable failures, not noise);
(2) cap the edit (diff-size budget in the prompt, validated structurally via text delta); (3) carry
a version only if its CV beats the parent's (already true for the pick, but the *chain* still walks
through regressions — consider improving always from the best-so-far version instead of the latest).

## D. Coverage backfill (cheap, parallel-safe with nothing else running)

- 21 taught-but-unverified symbols for the pick (run `acumen coverage --config config.yaml
  --tasks tasks_all_v2.working.yaml --skill v4 --skills skills_r2` to refresh the list — mostly
  `gr.calculate_niche*`, dataset loaders, `NhoodEnrichmentResult`).
- Targeted generation: `acumen tasks --per-notebook --notebook <substr>` for notebooks covering
  those symbols, or `--feedback "cover sq.gr.calculate_niche — write tasks whose answers exercise it"`;
  also the ~200 unrun `mined/` candidates (`--candidates mined --notebook <slug-substr>` to scope).
- New tasks merge into the WORKING set only (see `run_phase3.sh` merge block — it asserts lockbox
  disjointness). The lockbox is never regenerated.

## Ops checklist (hard-won; see lessons.md)

- One fleet OR one loop at a time — they share the session window.
- Long runs: `nohup caffeinate -i … &`; on `paused: … resets HH:MM` (exit 3) schedule
  `nohup zsh -c 'sleep <secs-past-reset>; <same command>' &`. Better: build the reset-time-aware
  retry (todo) instead of blind 30-min retries.
- Monitors: key liveness on `pgrep -f "acumen loop --config"` or a stored pid — a monitor's own
  command line matches `pgrep -f <script name>`.
- Never `cd ~/acumen-squidpy` in a shell that then writes `diary.md` or runs `git` — use
  `git -C /Users/tim.treis/Documents/GitHub/acumen` and absolute diary path.
- Diary: append per meaningful step (newest at bottom), commit+push; never commit a reformat of
  `src/acumen/_skills/data/references/python-api.md`.
- Suite: `.venv/bin/python -m pytest -q`; lint `uvx ruff@0.16.1 check src tests` + `format --check`.
