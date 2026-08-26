# acumen — master build plan ("build everything else")

Goal: finish the rulebook autoresearch loop. Optimize a versioned **rulebook** for squidpy, scored
on held-out tasks under **CV** with a **lockbox** (no selection leakage), over a **mined corpus**
that exhaustively covers the package, with a **multi-iteration loop** and a **size-not-cost**
leanness axis. Binding constraints: wall-clock + selection leakage, NOT dollar cost.

## Decisions (2026-08-26)
- **Execution:** sequential, one module at a time, each reviewed before the next.
- **Task supply:** build the full GitHub + web mining subsystem now (not just coverage-driven gen).
- **Model roles:** fable builds the mechanical modules; the CV/lockbox scientific core (P5/P7) stays
  on a strong model — its correctness is the whole point.

## The gate that sets the order
P5 (CV) over the current **3 tasks** is meaningless. The scientific core only pays off with a real
**corpus** (dozens+). So mining comes **before** P5. P3/P6/leanness/P4 are independent and cheap, so
they go first to de-risk and speed everything; mining builds the corpus; then P5, then P7.

---

## Sequence

### 1. P3 — warm dataset cache  ·  [fable]
**Why:** each bench sandbox re-downloads its squidpy dataset. Confirmed cause: squidpy writes to
`scanpy.settings.datasetdir` default `data` — **cwd-relative**, into the throwaway sandbox root.
**Build:**
- `config.py`: `dataset_cache_dirs: list[str]` (default `[]`; squidpy sets `["data", "cache"]`) — the
  target's cwd-relative download dirs.
- `sandbox.py`: before yielding, symlink each `root/<name>` → a persistent
  `cache_root/datasets/<cache_key>/<name>`. Isolation unchanged: acumen-owned dir, no env-scrub
  loosening, datasets are read-only inputs (no cross-run result leakage).
- `warm.py` + `acumen warm`: find the `sq.datasets.*` loaders referenced in the persisted P2 scripts
  (reuse `coverage.scan_references`), run each once **sequentially** in the target venv into the
  shared dir → parallel bench never races a download. Auto-invoke before the matrix in bench/loop.
**Risk:** concurrent first-download race — solved by pre-warm. **Tests:** symlinks created; warm
populates; second sandbox does no download (mocked loader).

### 2. P6-full — content-hashed rulebook artifact  ·  [fable]
**Why:** current `rulebooks.py` is a plain versioned file; skills are content-hashed + immutable with
a `meta.json` provenance chain. Bring rulebooks to parity.
**Build:** mirror `skills.py` — content hash, immutable dir, `meta.json` (parent version, rationale,
authoring model). Keep the `validate_rulebook` placeholder guard. Migrate `loop.py`/`rulebooks.py`
callers. **Tests:** hashing, immutability, provenance chain, resume.

### 3. Leanness Pareto axis = size, not cost  ·  [fable]
**Why:** cost is not an optimized metric; the report frontier is still cost-vs-success.
**Build:** `report.py` — replace the cost axis of the Pareto frontier with **skill size** (bytes of
SKILL.md + references). size-vs-success frontier. **Tests:** frontier picks the smaller skill at
equal success.

### 4. P4-full — difficulty strata pipeline + task selection  ·  [fable]
**Why:** `screen()` pools across models; strata must be per reference model and must *drive* which
tasks the loop uses.
**Build:** `difficulty.py` — `screen_by_model()` (no pooling), record the reference model;
`select_headroom_tasks(diffs, tasks)` → the tasks a skill can move. Wire selection into the loop's
task set. **Tests:** per-model strata; selection keeps only headroom.

### 5. Task mining subsystem  ·  [fable wiring, strong for design review]
**Why:** the intended corpus is mined real squidpy analyses at scale, not just the 50 bundled
notebooks. This is what makes P5 meaningful.
**Build (`mining.py` + `acumen mine`):**
- **GitHub harvest:** `gh` code search for real squidpy usage (`.py`/`.ipynb` importing squidpy,
  calling `sq.*`). Collect candidate analysis snippets. Respect rate limits; auth via `gh`.
- **Web harvest:** published tutorials/analyses beyond the bundled notebooks.
- **Candidates → tasks:** treat each mined snippet like a notebook shard — feed the existing sharded
  task-gen (`generate_tasks_sharded`) so mined analyses become validated tasks with persisted P2
  scripts (which then feed coverage + warm). Dedup by content hash + coverage-symbol overlap;
  quality-filter (runs? answer defensible? — reuses the analyst-defensible discipline).
- **Coverage-driven backfill:** use the P2 uncovered-symbol queue as `--feedback` to target gaps.
- **Output:** a corpus `tasks.yaml` of dozens–hundreds of tasks + `scripts/`.
**Risks:** GitHub rate limits/auth; web robustness; licensing (we keep goal+answer, not source —
note it); task quality variance (the quality filter + `acumen screen` gate it). **Tests:** harvest
parsing (mocked `gh`), dedup, candidate→shard adaptation.

### 6. P5 — CV over tasks + selection-leakage lockbox  ·  [STRONG]
**Why:** the load-bearing scientific piece. Today's train/test are within-task variants, not
held-out tasks — no real generalization signal, and leakage is possible.
**Build:**
- `folds.py`: `make_folds(tasks, k, seed)` — deterministic k-fold partition **by task** (sorted ids,
  hashed). Each fold: optimize-tasks vs held-out-tasks.
- **Lockbox:** a task subset set aside **before** CV that the optimize path structurally cannot read
  (separate file; the improve-agent guard denies reading lockbox tasks/answers, extending the
  existing test-split guard). Scored once, at the very end.
- Restructure `loop.py`: the rulebook is improved on optimize-fold run transcripts only, scored on
  held-out-fold tasks; report CV mean ± spread on PASS and LOAD. Improve agent never sees held-out
  or lockbox material — enforced by guard, not just prompt.
**Risk:** leakage is the whole point — every boundary (fold, lockbox) must be a *structural* guard
with a test that proves the agent is denied. **Tests:** fold determinism/partition; guard denies
held-out + lockbox reads; CV aggregation.

### 7. P7-full — multi-iteration loop + stopping rules  ·  [STRONG]
**Why:** current loop is a single v1→v2 pass.
**Build:** `loop.py` — iterate rulebook until a stopping rule (no CV improvement for N iterations, or
max-iterations, or wall-clock cap); select best version by held-out CV; then the one-shot lockbox
evaluation. Hard caps (iterations, wall-clock). **Tests:** stops on no-improvement; respects caps;
picks best-CV version; lockbox scored exactly once.

---

## Cross-cutting
- Working agreements: diary per module (newest at bottom, commit+push each step); pinned ruff 0.16.1
  (`uvx ruff@0.16.1 check/format --check src tests`); leave
  `src/acumen/_skills/data/references/python-api.md` alone; match codebase style (heavy "why"
  docstrings, structural guarantees over prompt instructions, per-unit result files, resume by file
  presence). Full suite green after each module.
- Model switching: switch to fable for modules 1–5 (mechanical); switch to a strong model for 6–7
  (the CV/lockbox core). The plan is model-independent; only authoring changes.
