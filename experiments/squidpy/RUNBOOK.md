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
| `mined/` | 282 candidate analyses mined from squidpy's notebooks. **~200 never turned into tasks.** |
| `rulebook/champion.md` | The best rulebook we found (r2-v4), with its rationale in `champion-meta.json`. |
| `results/lockbox_matrix.csv` | Per-task pass/fail for all 9 arms we benched. Our numbers, auditable. |
| `analyze.py` | Turns run trees into a verdict: draft means, paired sign tests, hard/easy split. |

---

## Setup

```bash
git clone https://github.com/timtreis/acumen && cd acumen
git checkout claude/acumen-overview-c410ls
uv sync                                   # or: pip install -e .
export ANTHROPIC_API_KEY=...              # then pass --auth api to every command below
cp -R experiments/squidpy ~/squidpy-exp && cd ~/squidpy-exp
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

36 held-out tasks cannot resolve a 5-task effect. ~200 mined candidates were never turned into
tasks; turning them into a second, larger, **never-opened** hold-out is the highest-value thing
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

```bash
acumen warm  --config config.yaml --tasks tasks_pool.yaml                       # pre-download datasets once
acumen bench --config config.yaml --tasks tasks_pool.yaml --runs runs \
  --no-skill --split test --auth api                                            # the floor; also feeds --headroom
```

Then score the champion rulebook we are handing you, as a mean over ≥3 drafts, on the *working*
pool only. This is the number Stage 3 has to beat, and it is the first honest baseline this
experiment has had.

```bash
acumen draft --config config.yaml --rulebook rulebook/champion.md --skills skills_d1 --auth api
acumen draft --config config.yaml --rulebook rulebook/champion.md --skills skills_d2 --auth api
acumen draft --config config.yaml --rulebook rulebook/champion.md --skills skills_d3 --auth api
for d in 1 2 3; do
  acumen bench --config config.yaml --tasks tasks_pool.yaml --runs runs_d$d \
    --skill v1 --skills skills_d$d --split test --auth api
done
python analyze.py noskill=runs/noskill champ.d1=runs_d1/skill_v1 \
  champ.d2=runs_d2/skill_v1 champ.d3=runs_d3/skill_v1
```

If the three drafts disagree by more than ~8pp, that is the expected draft noise and it sets
`--accept-delta` for Stage 3. **Write the measured spread down — it is a result in itself.**

## Stage 3 — evolve, for as many generations as you can afford

We ran 2 rounds of a 3-iteration loop. `evolve` is built for hundreds of generations and has never
been run live.

```bash
acumen evolve --config config.yaml --tasks tasks_pool.yaml \
  --rulebooks rulebooks --skills skills --runs runs_evolve \
  --islands 3 --generations 40 --headroom \
  --screen-size 24 --accept-delta <from stage 2> --confirm-every 3 \
  --no-lockbox --auth api --log-dir logs_evolve
```

- `--islands 3` evolves three rulebooks independently on disjoint task partitions, then
  cross-pollinates: only edits that replicated across ≥2 islands survive the merge. Replication
  across islands is the evidence standard — it is what makes a rule a *finding* rather than a fit.
- `--headroom` restricts evolution to tasks the bare model fails, so generations are not spent on
  tasks with no room to improve.
- `--confirm-every 3` tightens the ratchet: a screen win only becomes a champion after a
  full-benchmark confirmation, and a failed confirmation reverts. With compute to spare, confirm
  more often than we could.
- `--no-lockbox` keeps `lockbox2` sealed until Stage 4.
- Resume is free and agent-free: rerun the identical command after any interruption.

`runs_evolve/evolve.jsonl` is one line per generation — directive, screen subset, scores, decision.
That file is the dataset behind any claim about *what kind of edits* help, so keep it.

**Known limit, and the one place we would spend your compute if you have it to burn:** screens and
confirmations inside `evolve` score a candidate through a *single* draft, so selection is exposed
to the same ±8pp draft noise as everything else. `--accept-delta` is the blunt guard against it.
Averaging each candidate over N drafts at the screen step would remove the confound properly; it
is not implemented. If you have the budget, say so and we will build it — it is the experiment's
own conclusion turned into code.

## Stage 4 — the verdict, once

```bash
acumen bench --config config.yaml --tasks lockbox2/tasks.yaml --runs runs_verdict/noskill \
  --no-skill --split test --auth api
# then, for each of >=3 drafts of the merged champion and >=3 of the champion we gave you:
acumen bench --config config.yaml --tasks lockbox2/tasks.yaml --runs runs_verdict/<label> \
  --skill v1 --skills <draft skills root> --split test --auth api

python analyze.py noskill=runs_verdict/noskill \
  evolved.d1=... evolved.d2=... evolved.d3=... \
  handed.d1=... handed.d2=... handed.d3=...
```

Report what `analyze.py` prints, including when it says *NOT distinguishable from noise*. A clean
negative on a powered benchmark is a better result than the positive we currently have on an
under-powered one.

---

## What to send back

1. `runs_evolve/evolve.jsonl` — the generation-by-generation decision archive.
2. The final rulebooks: the merged champion plus each island's, with their `meta.json` rationales.
3. `analyze.py` output for Stages 2 and 4, verbatim.
4. `lockbox2/` (tasks + manifest) and the new `tasks_pool.yaml`, so the benchmark itself is shared.
5. Anything that broke. Platform failures that get recorded as task failures are the failure mode
   that has cost this experiment the most — one overnight network drop silently put an arm 9 tasks
   below its siblings and made a perfectly good skill look bad.

## Gotchas we paid for

- **A dead run is not a failed task.** Session limits, rate limits, overload and network drops are
  now all treated as transient: no `result.json` is written and the pass stops (exit code 3) rather
  than recording failures. Rerun to resume. If you see an arm that is inexplicably worse than its
  siblings, grep its `result.json` files for `"reason": "error"` before believing it.
- **Never run two agent fleets at once on one credential** — they contend, and on a subscription
  they trip the same window.
- **Agents are only as good as the ground truth.** Some task answers are genuinely ambiguous (a
  pair of cell-type names whose order is arbitrary). If a rulebook edit's rationale is "make the
  agent guess the ordering convention", that is the benchmark leaking, not a finding.
- **Skill size is a cost, not a win.** Our round-2 champion bought +2 tasks with +9 KB. Track size
  per draft; `acumen report` shows it.
