# Agent brief — continuing the squidpy skill-optimization experiment

You are continuing a measurement, not building a feature. `RUNBOOK.md` is the protocol: what to run,
in what order, and why. This file is how to operate it safely for days without a human watching.
Read both before running anything.

**The goal:** a powered verdict on one question — does an evolved rulebook beat (a) the bare model
and (b) the champion rulebook in `rulebooks/v1`, on a hold-out of at least 100 tasks that nothing
was selected on? Report it with `analyze.py`, whatever it says. A clean negative is a result.

## Setup

Needs git, [uv](https://docs.astral.sh/uv/), Python ≥3.12, network, ~20 GB disk (the squidpy checkout, its venv and datasets alone are ~12 GB), and an Anthropic API
key with a high rate limit. macOS or Linux.

```bash
git clone --branch claude/acumen-overview-c410ls https://github.com/timtreis/acumen ~/acumen
cd ~/acumen && uv sync && source .venv/bin/activate
cp -R experiments/squidpy ~/squidpy-exp && cd ~/squidpy-exp
export ANTHROPIC_API_KEY=...
python analyze.py --self-check                                   # prints "self-check ok"
acumen coverage --config config.yaml --tasks tasks_working.yaml  # clones squidpy, builds its venv; no tokens
```

Every command below runs from `~/squidpy-exp` with that venv active and `--auth api`.

## Hard rules

1. **Never look at `lockbox2/` early.** No `bench` on `lockbox2/tasks.yaml` and no
   `--lockbox lockbox2` on any run except the final evolve run. Each look spends its independence.
2. **Never edit a hold-out.** `lockbox/` and `lockbox2/` are write-once and digest-verified.
3. **Never change a flag on a resumed run.** Rerunning the identical command continues it; a
   changed command applies new decision rules on top of generations decided under the old ones.
   Start fresh `--runs/--skills/--rulebooks` trees instead.
4. **Never point two runs at one `--runs` tree**, and never run two agent-spawning `acumen`
   commands at once — they share your rate limit.
5. **Never touch answers or ground-truth scripts** to make a number move, and never paste hold-out
   task text into `--feedback`.
6. **Never delete a `result.json`** unless `status.py` reports it as platform-killed. Say that you did.

## Running long commands

Stage 1, the floor, and evolve each run for hours to days. Do not run them as a foreground tool
call — your harness will time out or background them unpredictably. Launch detached and poll:

```bash
nohup sh -c 'acumen evolve ...; echo "EXIT=$?"' > evolve.out 2>&1 &
echo $! > evolve.pid
```

When it is no longer alive, `tail -1 evolve.out` shows `EXIT=<code>`:

| Exit | Meaning | Do |
| --- | --- | --- |
| 0 | Finished | Next stage. |
| 3 | Paused: rate limit, overload or network drop. Nothing bad was recorded. | Wait 10 min, rerun the **identical** command. |
| 2 | Configuration or usage error | Read the message, fix, rerun. Do not retry blindly. |
| other | Crash | Read the last 50 lines of the output and the newest `logs_*/*.jsonl`. Stop and report. |

## Monitoring — every 20–30 minutes

```bash
kill -0 $(cat evolve.pid) && echo alive
python status.py runs          # progress per island, spend, skill-load rate; exit 1 = a PROBLEM line
tail -3 logs_evolve/*.jsonl    # the live step log, one JSON event per line
```

`status.py` checks the failure modes that have cost this experiment results before:

- **Platform-killed runs recorded as failures** — should be impossible now. If it appears, delete
  exactly those `result.json` files, rerun the identical command, and mention it in your report.
- **Accept rate above 50%** after 6 generations — `--accept-delta` is inside the noise. Stop the run
  and ask: raising it means restarting in fresh trees, which is a human call.
- **Zero accepts in 10 generations** — the bar may be out of reach. Same: stop and ask.
- **Skill load rate below 80%** — a draft may not trigger. Report which one.

Expect roughly $0.30 and 90 seconds per benchmark run. At the runbook's evolve settings that is on
the order of $25–35 and 30–40 minutes per generation, islands running one after another. Check
`status.py`'s spend line after the sanity run and extrapolate before launching the real one.

## Stop and ask a human when

- any stage ends with exit code 2 or a crash;
- Stage 1 yields fewer than 100 tasks in `lockbox2`;
- Stage 2's three champion drafts spread by more than 15 percentage points;
- `status.py` reports a PROBLEM you cannot resolve under the rules above;
- you want to change the protocol in any way — a different split, model, task filter or flag.

## Report

After each stage, a short note:

1. The exact commands you ran and their exit codes.
2. `status.py` and `analyze.py` output, **verbatim**.
3. Anything that broke, what you did about it, and any `result.json` you deleted.
4. At the end, the files listed under "What to send back" in `RUNBOOK.md`.
