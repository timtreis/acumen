# acumen — lessons (codebase gotchas)

Short rules discovered while working in this repo. Each has a Why.

- **Lint with the pinned ruff before committing.** `uvx ruff@0.16.1 check src tests` and
  `uvx ruff@0.16.1 format --check src tests`. `ruff format` reformats embedded code comments in
  `src/acumen/_skills/data/references/python-api.md` — that file is pre-existing and out of the
  hooks' scope; **leave it** (revert any change to it before committing).
  Why: the pre-commit hooks use a narrower scope than a manual `src tests` run; committing the
  markdown reformat is noise the maintainer didn't ask for.

- **Meta-agent modules share one isolation core; don't duplicate it.** `taskgen`, `draft`,
  `improve`, and `loop` all build a scrubbed env + PreToolUse guard + throwaway HOME/config. When
  adding a new meta-agent, reuse the pattern (or extract a helper like `taskgen._run_generation_agent`)
  rather than copy the env-scrub/guard — it's the security-sensitive part and two copies drift.
  Why: the env scrub is what keeps the operator's secrets out of a web-enabled agent.

- **Bench options must be identical across arms (baseline parity).** Anything added to
  `runner._build_options` (e.g. the sync-guard hook) must apply to both the noskill and skill arms.
  Why: the benchmark's validity rests on arms differing *only* by the presence of a skill dir.

- **The benchmark sandbox agent will background slow work and strand a one-shot run.** It has
  `run_in_background`/Monitor tools; on a slow task it defers and idles for a notification that never
  comes → `no_answer_file`. Mitigated by `runner.make_sync_guard`. Keep tasks inline-light too.
  Why: a stranded run scores a spurious failure that looks like a skill problem but isn't.

- **`skill_loaded` and `success` are orthogonal — track both.** A run can load the skill and still
  fail, or pass with no skill. A skill that never loads is a distinct, *prior* failure (only the
  frontmatter `description` governs loading). The loop's `Score` now carries both.
  Why: a 0/0 pass rate can be entirely a load failure; pass rate alone hides it.

- **squidpy's bundled datasets differ in normalization state; "top gene" tasks are fragile.**
  merfish is raw counts (robust), slideseqv2/visium are already log-normalized (re-normalizing flips
  the top gene). Many datasets have near-tied top genes. Graph *centrality* is deterministic (no
  perms) and cleaner, but centrality "most central" answers can be QC-artifact groups (`Low quality`)
  — not analyst-defensible. Verify ground-truth robustness before trusting a generated answer.
  Why: exact-match grading turns any answer fragility into a broken benchmark.

- **acumen reads auth from file/env, not the macOS Keychain.** A `claude` login on macOS lives in the
  login Keychain; acumen looks for `~/.claude/.credentials.json` or `CLAUDE_CODE_OAUTH_TOKEN`. Mint a
  portable token with `claude setup-token`. Note `setup-token` dumps its whole TUI to stdout when
  redirected — extract the `sk-ant-oat01-…` substring, don't use the raw capture.
  Why: `resolve_auth_mode` fails preflight otherwise, even on a machine that's "logged in".

- **A transient platform error must never become a `result.json`.** Session/usage/rate-limit and
  overload errors (`runner.is_transient`) are returned flagged, unwritten; `run_matrix` /
  `generate_tasks_sharded` pause (skip the queue) and raise `TransientLimitError` (CLI exit 3);
  the improve/draft agents map the same errors to the same exception.
  Why: the first live loop wrote 54 bogus "failures" in minutes when the limit hit — resume would
  have kept them and the improver would have read them as evidence about the skill.

- **Never write an extra file into a version directory after `meta.json`.** The content hash covers
  every file except `meta.json`; a diff or note dropped in later makes the version read as tampered
  on the next load (rulebook diffs live in `rulebooks/diffs/`, `rulebooks.diff_path`).
  Why: round 1's multi-iteration resume died on exactly this; single-iteration runs never reload a
  version, so the bug was invisible until then.

- **The Claude Code harness backgrounds a Bash call that outlives its timeout — no tool flag to
  deny.** Isolated agents get `BASH_DEFAULT_TIMEOUT_MS`/`BASH_MAX_TIMEOUT_MS` = 45 min in
  `env.scrubbed_env`; the PreToolUse sync guard only covers the agent *choosing* to background.
  Why: shards stalled "waiting for a notification" with zero denied tool calls.

- **Agent-written YAML frontmatter is normalized, not failed.** An unquoted colon in `description:`
  is the agent's content with our formatting constraint; `skills.normalize_frontmatter` re-quotes
  (only when parsing fails) before validation. Why: a fold draft killed a whole loop run over quoting.

- **`asyncio.as_completed` launches in arbitrary order — never write order-dependent tests over it.**
  Key fakes off call-count, not task id. Why: two pause tests failed on this exact assumption.

- **`pgrep -f <pattern>` inside a Monitor matches the monitor's own command line.** Key liveness on
  the concrete worker process (e.g. `acumen loop --config`) or a stored pid, never on the wrapper
  script's name. Why: monitors reported RUNNING long after the run died.
