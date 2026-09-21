# squidpy skill-optimization experiment — collaborator runbook

**The question:** can an agent optimize a skill for a Python package in a way that provably
generalizes to tasks nobody selected on?

**Where we got to:** on a 36-task held-out lockbox, the no-skill floor is 22/36 (61%). The best
rulebook we produced scores a **mean of 27/36 (75%) over drafts** — but individual drafts of that
same rulebook score 29 and 25, and a paired sign test clears p<0.05 for only one of them. The
effect is probably real and definitely under-powered.

**What we need from you:** compute. Every number above is limited by a 36-task benchmark and by a
selection signal that was measured once per version. Both are fixable by spending tokens, which is
what you have and we did not.

Two facts to carry through everything below:

1. **The unit of measurement is the draft, not the task.** The same rulebook text drafted into a
   skill twice scores differently (measured sd ≈ 3 of 36 tasks ≈ 8pp). Draft noise is a per-skill
   offset — running *more tasks* does not shrink it, only *more drafts* do.
2. **A skill can make things worse.** On the 22 lockbox tasks the bare model already passes, our
   drafts kept 18–22. Report regressions separately from gains; an overall rate hides them.

---

## What's in this directory

| Path | What it is |
| --- | --- |
| `config.yaml` | Target (squidpy), models, concurrency. Already raised to 16 — see Setup. |
| `tasks_working.yaml` | 228 tasks with ground-truth answers. The training/selection pool. |
| `lockbox/` | 36 held-out tasks + digest manifest. **Already opened 9 times** — see Stage 2. |
| `scripts/` | 260 executable ground-truth scripts, one per task, that produce the expected answers. |
| `mined/` | The 200 mined candidate analyses that were **never turned into tasks** — Stage 1's input. |
| `mined_used/` | The 81 that were, kept for provenance. `mined/index.json` records every origin URL. |
| `rulebooks/v1/` | The best rulebook we found (r2-v4), pre-seeded as the chain's `v1` so `evolve` and every island start from it. Its rationale traces the lineage. |
| `results/lockbox_matrix.csv` | Per-task pass/fail for all 9 arms we benched. Our numbers, auditable. |
| `analyze.py` | Turns run trees into a verdict: draft means, paired sign tests, hard/easy split. |
| `status.py` | One-screen health check for a running experiment. Exit 1 when something needs a human. |
| `AGENTS.md` | Operating rules for an agent running this unattended: launching, monitoring, when to stop. |

---

## Setup

```bash
git clone --branch claude/acumen-overview-c410ls https://github.com/timtreis/acumen ~/acumen
cd ~/acumen && uv sync && source .venv/bin/activate    # puts `acumen` on PATH
cp -R experiments/squidpy ~/squidpy-exp && cd ~/squidpy-exp
export ANTHROPIC_API_KEY=...                           # then pass --auth api to every command below
```

**Use `--auth api`, not a subscription.** Everything in this experiment that went wrong
operationally came from subscription session windows: runs pausing mid-benchmark, blind retry
wrappers, rounds spread over days. On API credit that entire class of problem disappears and
`max_concurrency` becomes the only limit. We ran at 2. Raise it as far as your rate limit allows.

Sanity check before spending anything:

```bash
acumen coverage --config config.yaml --tasks tasks_working.yaml    # expect API coverage 30/99
python analyze.py --self-check
```

---

## Stage 1 — grow the benchmark (do this first)

36 held-out tasks cannot resolve a 5-task effect. The 200 candidates in `mined/` were never turned
into tasks, and none shares a source notebook with an existing task — so a hold-out carved from them
is clean. Turning them into a second, larger, **never-opened** hold-out is the highest-value thing
compute can buy here.

```bash
# 1a. Candidates -> tasks. One agent per candidate, sharded and resumable: delete a shard file to
#     redo just that one. Expect a large fraction to be discarded as ungradeable; that is correct.
acumen tasks --config config.yaml --candidates mined --out tasks_new.yaml \
  --shards-dir tasks_new.shards --auth api --log-dir logs_new

# 1b. Split the new tasks: half into a fresh hold-out, half into the working pool.
acumen lockbox --tasks tasks_new.yaml --out lockbox2 --working tasks_new.working.yaml --fraction 0.5
```

```bash
# 1c. Merge the new working tasks into the pool, asserting both hold-outs stay disjoint from it.
python - <<'EOF'
import yaml
from pathlib import Path
from acumen.tasks import parse_tasks
from acumen.taskgen import dump_tasks
from acumen.folds import read_lockbox

held = set(read_lockbox(Path("lockbox")).task_ids) | set(read_lockbox(Path("lockbox2")).task_ids)
merged = []
for f in ("tasks_working.yaml", "tasks_new.working.yaml"):
    ts = parse_tasks(yaml.safe_load(Path(f).read_text()))
    print(f"  {f}: {len(ts)}")
    merged += ts
tasks = parse_tasks(yaml.safe_load(dump_tasks(merged)))
assert not ({t.id for t in tasks} & held), "a working task is in a hold-out"
Path("tasks_pool.yaml").write_text(dump_tasks(tasks))
print(f"  tasks_pool.yaml: {len(tasks)} working, {len(held)} held out")
EOF

acumen coverage --config config.yaml --tasks tasks_pool.yaml   # how much of the API this now verifies
```

**Target:** ≥100 tasks in `lockbox2`. At 100 paired tasks a +5pp effect is detectable; at 36 it is
not. Stop Stage 1 when `lockbox2` is big enough, not when the candidates run out.

`lockbox2` is write-once and digest-verified. **Do not benchmark against it until Stage 3 is
finished** — every look costs independence. Our original `lockbox/` has been opened 9 times and
should now be treated as a validation set, not a clean hold-out.

## Stage 2 — re-establish the floor, over drafts

**Every command from here on uses the one run tree `runs/`.** `evolve --headroom` reads the
no-skill floor from its own `--runs` root; a floor benched anywhere else is invisible to it.

```bash
acumen warm  --config config.yaml --tasks tasks_pool.yaml          # pre-download datasets once
acumen bench --config config.yaml --tasks tasks_pool.yaml --runs runs \
  --no-skill --split test --auth api                               # the floor; feeds --headroom
```

Then score the champion we are handing you as a mean over 3 drafts, on the *working* pool only.
This is the number Stage 3 has to beat, and the first honest baseline this experiment has had.

```bash
for d in 1 2 3; do
  acumen draft --config config.yaml --rulebook rulebooks/v1/rulebook.md --skills champ_d$d --auth api
  acumen bench --config config.yaml --tasks tasks_pool.yaml --runs runs_champ_d$d \
    --skill v1 --skills champ_d$d --split test --auth api
done
python analyze.py noskill=runs/noskill champ.d1=runs_champ_d1/skill_v1 \
  champ.d2=runs_champ_d2/skill_v1 champ.d3=runs_champ_d3/skill_v1
```

The per-draft spread this prints sets `--accept-delta` for Stage 3: roughly half the spread in
passes-per-screen, and never below 2. **Write the spread down — it is a result in itself.**

## Stage 3 — evolve, and the verdict

We ran 2 rounds of a 3-iteration loop. `evolve` is built for hundreds of generations and has never
been run live. **First, a throwaway sanity run in its own trees, with the hold-out sealed:**

```bash
cp -R rulebooks rulebooks_sanity
acumen evolve --config config.yaml --tasks tasks_pool.yaml \
  --rulebooks rulebooks_sanity --skills skills_sanity --runs runs_sanity \
  --generations 2 --screen-size 12 --screen-drafts 3 --accept-delta 2 \
  --no-lockbox --auth api --log-dir logs_sanity
```

Check `runs_sanity/evolve.jsonl` has two lines whose screen `total` is 36 (12 tasks x 3 drafts),
then delete every `*_sanity` tree. **Never point a second run at an existing `--runs` tree:** results
are resumed by path, not by skill, so a run that finds `runs/skill_v1/` already populated reuses
those results even if they came from a different draft.

**Then the real run.** It evolves, merges the islands, validates the merge on the full working
pool, and only then opens `lockbox2` — once — scoring the seed and the merged champion over 3
drafts each:

```bash
acumen evolve --config config.yaml --tasks tasks_pool.yaml \
  --rulebooks rulebooks --skills skills --runs runs \
  --islands 3 --generations 40 --headroom \
  --screen-size 24 --screen-drafts 3 --accept-delta <from stage 2> --confirm-every 3 \
  --lockbox lockbox2 --drafts 3 --auth api --log-dir logs_evolve
```

- `--screen-drafts 3` is the flag this experiment's own findings paid for: each candidate is
  drafted three times and judged on the sum, so a generation is accepted for signal rather than
  draft luck. `--accept-delta` still means *passes per draft*; the bar scales automatically.
- `--islands 3` evolves three rulebooks independently on disjoint task partitions, each starting
  from `rulebooks/v1`, then cross-pollinates: only edits that replicated across ≥2 islands
  survive the merge. Replication across islands is the evidence standard.
- `--headroom` restricts evolution to tasks the bare model fails.
- `--confirm-every 3` tightens the ratchet: a screen win only becomes a champion after a
  full-benchmark confirmation, and a failed confirmation reverts.
- Resume is free and agent-free: rerun the **identical** command after any interruption. Changing a
  flag mid-run changes what is being measured — start fresh trees instead.

Each island keeps one line per generation in `runs/islands/island-<i>/evolve.jsonl` — directive,
screen subset, scores, decision. Those files are the dataset behind any claim about *what kind of
edits* help, so keep them.

When it finishes it prints `merged into vN`. Bench the bare model on the same hold-out, then run
the verdict:

```bash
acumen bench --config config.yaml --tasks lockbox2/tasks.yaml --runs runs/lockbox \
  --no-skill --split test --auth api
python analyze.py noskill=runs/lockbox/noskill \
  seed.d1=runs/lockbox/skill_v1 seed.d2=runs/drafts/v1/d2/lockbox/skill_v1 \
  seed.d3=runs/drafts/v1/d3/lockbox/skill_v1 \
  merged.d1=runs/lockbox/skill_vN merged.d2=runs/drafts/vN/d2/lockbox/skill_v1 \
  merged.d3=runs/drafts/vN/d3/lockbox/skill_v1
```

Report what `analyze.py` prints, including when it says *NOT distinguishable from noise*. A clean
negative on a powered benchmark is a better result than the positive we currently have on an
under-powered one.

---

## What to send back

1. `runs/islands/island-*/evolve.jsonl` — the generation-by-generation decision archives.
2. `rulebooks/` — the merged champion (its `meta.json` rationale is the meta-rule list) and each
   island's chain under `rulebooks/islands/`.
3. `analyze.py` output for Stage 2 and the verdict, verbatim.
4. `lockbox2/` (tasks + manifest) and `tasks_pool.yaml`, so the benchmark itself is shared.
5. Anything that broke. Platform failures recorded as task failures are the failure mode that has
   cost this experiment the most — one overnight network drop silently put an arm 9 tasks below its
   siblings and made a perfectly good skill look bad.

## Gotchas we paid for

- **A dead run is not a failed task.** Session limits, rate limits, overload and network drops are
  now all treated as transient: no `result.json` is written and the pass stops (exit code 3) rather
  than recording failures. Rerun to resume. `python status.py runs` flags any platform-killed run
  that was recorded anyway; run it before believing an arm that is inexplicably worse than its
  siblings.
- **Never run two agent fleets at once on one credential** — they contend for the same rate limit.
- **Agents are only as good as the ground truth.** Some task answers are genuinely ambiguous (a
  pair of cell-type names whose order is arbitrary). If a rulebook edit's rationale is "make the
  agent guess the ordering convention", that is the benchmark leaking, not a finding.
- **Skill size is a cost, not a win.** Our round-2 champion bought +2 tasks with +9 KB. Track size
  per draft; `acumen report` shows it.
