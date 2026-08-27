"""Command-line entry point — a thin shell over the importable API."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path

from acumen.bench import build_matrix, pending, run_matrix, summarize
from acumen.config import Config, ConfigError, load_config
from acumen.coverage import CoverageError, inventory_in_venv, load_scripts, measure_coverage, skill_mentions
from acumen.difficulty import HeadroomSelection, screen
from acumen.draft import DraftError, draft_skill
from acumen.env import DEFAULT_CACHE_ROOT, AuthMode, EnvError, Target, prepare_target, resolve_auth_mode
from acumen.folds import FoldError, split_lockbox, write_lockbox
from acumen.improve import ImproveError, improve_skill
from acumen.logs import LiveLog
from acumen.loop import CVResult, LoopError, StopRule, run_iteration, run_loop
from acumen.mining import (
    _KNOWN_ALIASES,
    SEARCH_INTERVAL_S,
    MineResult,
    MiningError,
    _repo_id,
    default_queries,
    mine_github,
    mine_urls,
    submodule_repos,
    write_candidates,
)
from acumen.paths import SPLITS
from acumen.report import ReportError, build_report
from acumen.rulebooks import RulebookError
from acumen.runner import RunOutcome, StderrFilter, TransientLimitError
from acumen.scaffold import InitError, scaffold
from acumen.ship import ShipError, ship_skill
from acumen.skills import SkillError, available_versions, latest_version, load_skill
from acumen.taskgen import (
    SCRIPTS_DIRNAME,
    ShardOutcome,
    TaskGenError,
    dump_tasks,
    generate_tasks,
    generate_tasks_sharded,
)
from acumen.tasks import TaskError, load_tasks
from acumen.warm import WarmOutcome, collect_dataset_calls, warm_datasets


def _add_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    parser.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    arm = parser.add_mutually_exclusive_group()
    arm.add_argument("--no-skill", action="store_true", help="run the baseline arm (the default)")
    arm.add_argument("--skill", metavar="VERSION", help="run with a skill version, e.g. v1")
    parser.add_argument("--split", choices=SPLITS, action="append", help="restrict to a split (repeatable)")
    parser.add_argument("--task", metavar="ID", action="append", help="restrict to a task id (repeatable)")
    parser.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    parser.add_argument("--replicates", type=int, help="override config n_replicates")
    parser.add_argument("--no-resume", action="store_true", help="re-run runs that already completed")
    parser.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    parser.add_argument("--keep-sandboxes", action="store_true", help="leave run sandboxes on disk")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    parser.add_argument("--no-warm", action="store_true", help="skip pre-downloading datasets into the shared cache")
    parser.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix and exit without running agents")


def _add_log_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared live-log flags to a meta-agent subcommand."""
    parser.add_argument("--stream", action="store_true", help="mirror the agent's conversation to the terminal live")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("logs"), dest="log_dir", help="directory for the run log (default: logs/)"
    )


def _add_feedback_arg(parser: argparse.ArgumentParser, *, extra: str = "") -> None:
    """Add the optional ``--feedback`` flag to an authoring subcommand.

    The text is injected into the agent's prompt as a subordinated guidance block; it never
    overrides the isolation or anti-overfit rules. ``extra`` appends a per-command note to the
    help text.
    """
    help_text = "extra guidance for the agent, injected into its prompt as subordinate guidance"
    parser.add_argument("--feedback", help=(help_text + extra) or None)


def _add_auth_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--auth`` flag to a subcommand.

    Every agentic command defaults to the Claude subscription ("session") when a subscription
    login is present and falls back to the API key otherwise. ``bench`` carries this flag too —
    it used to be barred from the subscription to keep its recorded ``cost_usd`` real, but cost is
    not a metric acumen optimizes, so a subscription ``bench`` is allowed and simply records no
    meaningful per-run cost.
    """
    parser.add_argument(
        "--auth",
        choices=("auto", "session", "api"),
        default="auto",
        help="which credential to bill: 'session' (Claude subscription), 'api' (Anthropic API), "
        "or 'auto' (default: session if you're logged in, else the API)",
    )


def _print_auth(mode: AuthMode) -> None:
    """Report which credential the run will bill, so the choice is never silent."""
    label = "Claude subscription (session)" if mode == "session" else "API key"
    print(f"auth: {label}", flush=True)


def _print_log_result(log: LiveLog) -> None:
    """Print where the rendered HTML log landed, once a run has finalized."""
    if log.html_rendered:
        print(f"log → {log.html_path}")
    else:
        print("note: HTML log not rendered (claude-code-log missing?) — the jsonl log is complete", file=sys.stderr)


def _fmt_secs(seconds: float) -> str:
    """Compact wall-clock duration, e.g. ``9s`` / ``2m41s`` / ``1h04m``."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _key_label(key) -> str:
    return f"{key.arm}/{key.split}/{key.model}/{key.task_id}/rep_{key.rep}"


class _Progress:
    """Progress reporter for a concurrent bench pass.

    Prints a line when each run starts and finishes, each stamped with the wall-clock
    elapsed since the pass began, the number in flight, and a running pass tally — the
    context a long, interleaved pass needs to be readable as it scrolls by.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.started = 0
        self.done = 0
        self.passed = 0
        self.running = 0
        self._t0 = time.monotonic()

    @property
    def elapsed(self) -> float:
        """Seconds since the pass began."""
        return time.monotonic() - self._t0

    def _stamp(self) -> str:
        return f"+{_fmt_secs(self.elapsed):>6}"

    def on_start(self, item) -> None:
        self.started += 1
        self.running += 1
        print(
            f"[{self._stamp()}] ▶ start {_key_label(item.key)}"
            f"  (running {self.running}, {self.started}/{self.total} started)",
            flush=True,
        )

    def on_done(self, outcome: RunOutcome) -> None:
        self.done += 1
        self.running -= 1
        if outcome.success:
            self.passed += 1
        mark = "✓ pass" if outcome.success else "✗ FAIL"
        p = outcome.payload
        toks = int(p.get("input_tokens", 0)) + int(p.get("output_tokens", 0))
        cost = float(p.get("cost_usd", 0.0))
        dur = _fmt_secs(float(p.get("duration_s", 0.0)))
        stats = f"{_fmt_tokens(toks)}tok ${cost:.2f} {dur}"
        print(
            f"[{self._stamp()}] {mark} {_key_label(outcome.key)}"
            f"  ({outcome.reason})  {stats}"
            f"  [{self.done}/{self.total} done, {self.passed} passed]",
            flush=True,
        )


def _fmt_tokens(value: int) -> str:
    """Compact token count, e.g. ``118k`` / ``1.2M``."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return str(value)


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)
    if args.replicates:
        cfg = replace(cfg, n_replicates=args.replicates)

    version = args.skill  # None => the noskill baseline
    skill = None
    if version is not None:
        skill = load_skill(args.skills, version, expect_name=cfg.skill_name)

    planned = build_matrix(cfg, tasks, skill=version, splits=args.split or SPLITS, task_ids=args.task)
    todo = pending(planned, args.runs, resume=not args.no_resume)

    arm = "noskill" if version is None else f"skill_{version}"
    print(f"arm {arm}: {len(planned)} runs planned, {len(planned) - len(todo)} already complete, {len(todo)} to run")
    if skill is not None:
        print(f"skill {skill.version}: {skill.name} ({skill.hash[:19]}…)")
    if args.dry_run:
        for item in todo:
            k = item.key
            print(f"  {k.arm}/{k.split}/{k.model}/{k.task_id}/rep_{k.rep}")
        return 0
    if not todo:
        return 0

    auth_mode = resolve_auth_mode(args.auth)
    _print_auth(auth_mode)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]} (venv {target.venv_dir})", flush=True)
    if cfg.dataset_cache_dirs and not args.no_warm:
        _warm_cache(cfg, target, args.tasks)

    print(f"running {len(todo)} runs, up to {cfg.max_concurrency} at a time:", flush=True)
    progress = _Progress(len(todo))
    outcomes = asyncio.run(
        run_matrix(
            todo,
            target=target,
            runs_root=args.runs,
            max_concurrency=cfg.max_concurrency,
            auth_mode=auth_mode,
            skill=skill,
            skill_name=cfg.skill_name,
            keep_sandbox=args.keep_sandboxes,
            stderr=StderrFilter(),
            on_start=progress.on_start,
            on_done=progress.on_done,
            env_passthrough=cfg.env_passthrough,
            dataset_cache_dirs=cfg.dataset_cache_dirs,
        )
    )

    passed = sum(1 for o in outcomes if o.success)
    counts = summarize(outcomes)
    breakdown = ", ".join(f"{reason}={n}" for reason, n in sorted(counts.items()))
    total_cost = sum(float(o.payload.get("cost_usd", 0.0)) for o in outcomes)
    print(f"\n{passed}/{len(outcomes)} passed in {_fmt_secs(progress.elapsed)}  (${total_cost:.2f}, {breakdown})")

    # The comparison is only meaningful if the skill actually reached the agent, so say
    # so rather than leaving it to be discovered later in the transcripts.
    loaded = sum(1 for o in outcomes if o.payload.get("skill_loaded"))
    if skill is not None:
        print(f"skill loaded in {loaded}/{len(outcomes)} runs")
        if loaded == 0:
            print(
                "warning: the skill never loaded — this arm is not measuring the skill",
                file=sys.stderr,
            )
    elif loaded:
        print(f"warning: {cfg.skill_name} loaded in {loaded} baseline runs", file=sys.stderr)
    print(f"runs written to {args.runs.resolve()}")
    return 0


def _cmd_draft(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, draft_model=args.model)

    existing = available_versions(args.skills)
    if existing and not args.force:
        print(
            f"skills already exist ({', '.join(existing)}) — drafting would add "
            f"another version. Pass --force to draft anyway, or use `acumen improve` "
            f"to build on {existing[-1]}.",
            file=sys.stderr,
        )
        return 2

    auth_mode = resolve_auth_mode(args.auth)
    _print_auth(auth_mode)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(f"drafting with {cfg.draft_model} (this reads the package source) ...", flush=True)

    log = LiveLog.open(args.log_dir, "draft", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            draft_skill(
                cfg=cfg,
                target=target,
                skills_root=args.skills,
                auth_mode=auth_mode,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                feedback=args.feedback,
                log=log,
            )
        )
    skill = result.skill
    files = sorted(p.relative_to(skill.directory).as_posix() for p in skill.directory.rglob("*") if p.is_file())
    print(f"\nwrote {skill.directory}")
    print(f"  name:        {skill.name}")
    print(f"  description: {skill.description}")
    print(f"  hash:        {skill.hash}")
    print(f"  files:       {', '.join(files)}")
    print(f"  cost:        ${result.cost_usd:.2f} over {result.turns} turns")
    _print_log_result(log)
    print(f"\nnext: acumen bench --skill {skill.version}")
    return 0


def _cmd_improve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.model:
        cfg = replace(cfg, improve_model=args.model)

    versions = available_versions(args.skills)
    if not versions:
        print(
            f"no skill versions under {args.skills} — run `acumen draft` first, then bench it",
            file=sys.stderr,
        )
        return 2
    parent = args.from_version or latest_version(args.skills)
    # Immutability guard: the improved version is always the next unused directory,
    # so an existing version is never in the write path. Say the parent plainly up front.
    skill = load_skill(args.skills, parent, expect_name=cfg.skill_name)
    print(f"improving {skill.version} ({skill.name}, {skill.hash[:19]}…) with {cfg.improve_model}")

    auth_mode = resolve_auth_mode(args.auth)
    _print_auth(auth_mode)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)

    log = LiveLog.open(args.log_dir, "improve", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            improve_skill(
                cfg=cfg,
                target=target,
                skills_root=args.skills,
                runs_root=args.runs,
                tasks=tasks,
                auth_mode=auth_mode,
                parent_version=parent,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                feedback=args.feedback,
                log=log,
            )
        )
    new = result.skill
    files = sorted(p.relative_to(new.directory).as_posix() for p in new.directory.rglob("*") if p.is_file())
    print(f"\nwrote {new.directory}  (parent {result.parent})")
    print(f"  name:        {new.name}")
    print(f"  description: {new.description}")
    print(f"  hash:        {new.hash}")
    print(f"  files:       {', '.join(files)}")
    print(f"  evidence:    {result.n_train_runs} train runs ({result.n_train_failures} failing)")
    print(f"  cost:        ${result.cost_usd:.2f} over {result.turns} turns")
    if new.hash == skill.hash:
        print(
            "warning: the new version is byte-identical to its parent — the improver changed nothing",
            file=sys.stderr,
        )
    _print_log_result(log)
    print(f"\nnext: acumen bench --skill {new.version} && acumen report")
    return 0


def _cmd_tasks(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, tasks_model=args.model)

    out = args.out
    # Fail before the (costly) target prep and agent run if we'd have to clobber.
    if out.exists() and not args.force:
        print(
            f"{out} already exists — pass --force to overwrite it (e.g. the placeholder from `acumen init`)",
            file=sys.stderr,
        )
        return 2

    auth_mode = resolve_auth_mode(args.auth)
    _print_auth(auth_mode)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)

    if args.per_notebook or args.candidates:
        return _cmd_tasks_sharded(args, cfg, target, out, auth_mode)

    print(
        f"generating tasks with {cfg.tasks_model} (this reads the source and runs package code) ...",
        flush=True,
    )

    log = LiveLog.open(args.log_dir, "tasks", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            generate_tasks(
                cfg=cfg,
                target=target,
                out_path=out,
                auth_mode=auth_mode,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                force=args.force,
                feedback=args.feedback,
                log=log,
            )
        )
    print(f"\nwrote {result.out_path.resolve()}")
    print(f"  tasks: {len(result.tasks)} ({', '.join(t.id for t in result.tasks)})")
    print(f"  cost:  ${result.cost_usd:.2f} over {result.turns} turns")
    _print_log_result(log)
    print("\nnext: review the tasks, then `acumen draft` and `acumen bench`")
    return 0


def _cmd_tasks_sharded(args: argparse.Namespace, cfg: Config, target: Target, out: Path, auth_mode: AuthMode) -> int:
    """Run sharded task generation (per notebook, or per mined candidate), streaming progress."""
    shards_dir = args.shards_dir or out.parent / f"{out.stem}.shards"
    unit = "mined candidate" if args.candidates else "notebook"
    print(
        f"generating tasks per {unit} with {cfg.tasks_model}, up to {cfg.max_concurrency} at once "
        f"(each reads its {unit} and runs package code) ...",
        flush=True,
    )
    print(f"shards → {shards_dir}   logs → {args.log_dir}", flush=True)

    def on_done(outcome: ShardOutcome) -> None:
        if outcome.status == "cached":
            mark, detail = "·", f"cached, {outcome.n_tasks} tasks"
        elif outcome.status == "generated":
            mark, detail = "✓", f"{outcome.n_tasks} tasks, ${outcome.cost_usd:.2f}"
        else:
            mark, detail = "✗", f"FAILED — {outcome.error}"
        print(f"  {mark} {outcome.slug}: {detail}", flush=True)

    result = asyncio.run(
        generate_tasks_sharded(
            cfg=cfg,
            target=target,
            out_path=out,
            shards_dir=shards_dir,
            auth_mode=auth_mode,
            max_turns=args.max_turns,
            max_usd=args.max_usd,
            force=args.force,
            feedback=args.feedback,
            log_dir=args.log_dir,
            stream=args.stream,
            notebook_filter=args.notebook,
            candidates_dir=args.candidates,
            on_shard_done=on_done,
        )
    )
    print(f"\nwrote {result.out_path.resolve()}")
    print(f"  tasks:  {len(result.tasks)} from {result.n_ok}/{len(result.outcomes)} shards")
    if result.n_failed:
        failed = ", ".join(o.slug for o in result.outcomes if o.status == "failed")
        print(f"  failed: {result.n_failed} shards ({failed}) — rerun to retry them", file=sys.stderr)
    print(f"  cost:   ${result.cost_usd:.2f}")
    print("\nnext: review the tasks, then `acumen draft` and `acumen bench`")
    return 0


def _cmd_screen(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    diffs = screen(args.runs, tasks, splits=args.split or SPLITS, by_model=args.by_model)
    if not diffs:
        print(
            f"no baseline runs found under {args.runs}/noskill — run `acumen bench --no-skill "
            f"--tasks {args.tasks}` first, then screen",
            file=sys.stderr,
        )
        return 2
    print("baseline difficulty (a task with headroom is one the baseline does NOT already pass):")
    model_col = f" {'model':<28}" if args.by_model else ""
    print(f"  {'task':<44} {'split':<6}{model_col} {'baseline':<10} stratum")
    headroom: list[str] = []
    for d in diffs:
        model_cell = f" {d.model or '':<28}" if args.by_model else ""
        print(f"  {d.task_id:<44} {d.split:<6}{model_cell} {d.passed}/{d.total} ({d.pass_rate:.0%})   {d.stratum}")
        if d.has_headroom:
            headroom.append(f"{d.task_id}[{d.split}]" + (f"@{d.model}" if d.model else ""))
    if headroom:
        print(f"\ntasks with headroom for the loop: {', '.join(headroom)}")
    else:
        print("\nno task has headroom — the baseline already passes them all; the loop cannot show movement")
    return 0


def _warm_cache(cfg: Config, target: Target, tasks_path: Path) -> list[WarmOutcome]:
    """Pre-download every dataset the tasks' ground-truth scripts load into the shared cache.

    Runs before a benchmark matrix so concurrent runs never race the same first download. Only
    meaningful when ``config.dataset_cache_dirs`` is set — without the sandbox symlinks the shared
    cache is never seen — so callers gate on that. Reads the same ``scripts/`` dir as coverage.
    """
    scripts = load_scripts(tasks_path.parent / SCRIPTS_DIRNAME)
    calls = collect_dataset_calls(scripts, target.pkg_name)
    if not calls:
        print(f"warm: no dataset loader calls found in {tasks_path.parent / SCRIPTS_DIRNAME} — nothing to pre-download")
        return []
    print(f"warming {len(calls)} dataset(s) into {target.datasets_dir} (sequential, once per target):", flush=True)

    def on_done(outcome: WarmOutcome) -> None:
        mark = "✓" if outcome.ok else "✗"
        print(f"  {mark} {outcome.call.source}" + (f"  ({outcome.error})" if not outcome.ok else ""), flush=True)

    return warm_datasets(target.python, target.pkg_name, calls, target.datasets_dir, on_done=on_done)


def _cmd_warm(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    if not cfg.dataset_cache_dirs:
        print(
            "warning: config has no 'dataset_cache_dirs' — sandboxes will not see the shared cache, so "
            "warming it has no effect on benchmark runs. Set e.g. dataset_cache_dirs: [data, cache].",
            file=sys.stderr,
        )
    outcomes = _warm_cache(cfg, target, args.tasks)
    failed = [o for o in outcomes if not o.ok]
    return 1 if failed else 0


def _cmd_mine(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    # The gate counts only the package's real public API as a "symbol", the same inventory
    # coverage uses — so a mined file is kept for touching real analyses, not for mentioning names.
    inventory = inventory_in_venv(target.python, target.pkg_name).names
    results = []
    if not args.no_github:
        queries = args.query or default_queries(target.pkg_name, args.alias)
        # The target itself, its tutorial submodules (already sharded as notebooks), and anything
        # the operator names: not analyses, or already covered.
        own = _repo_id(cfg.repo)
        exclude = [*([own] if own else []), *submodule_repos(target.src_dir), *(args.exclude_repo or [])]
        print(
            f"GitHub code search: {len(queries)} queries x (.ipynb, .py), up to {args.limit} hits each, "
            f"paced at {SEARCH_INTERVAL_S:.0f}s (10 searches/min limit) ...",
            flush=True,
        )
        results.append(
            mine_github(
                target.pkg_name,
                queries,
                exclude_repos=exclude,
                limit=args.limit,
                inventory=inventory,
                min_symbols=args.min_symbols,
                on_progress=lambda msg: print(f"  {msg}", flush=True),
            )
        )
    if args.url:
        print(f"web pages: {len(args.url)} ...", flush=True)
        results.append(mine_urls(target.pkg_name, args.url, inventory=inventory, min_symbols=args.min_symbols))

    written, rejected = write_candidates(results, args.out)
    kept = sum(len(r.kept) for r in results)
    print(f"\nmined {kept} candidate(s) ({len(written)} new) → {args.out.resolve()}")
    for cand in (c for r in results for c in r.kept):
        print(f"  + {cand.slug}  [{cand.kind}] {len(cand.symbols)} symbols  {cand.origin}")
    if rejected:
        print(f"\nrejected {len(rejected)}:")
        for reason, n in MineResult(rejected=rejected).reasons().items():
            print(f"  {n:4d}  {reason}")
    print(f"\nnext: `acumen tasks --candidates {args.out} --out tasks_mined.yaml` to turn them into tasks")
    return 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)

    inventory = inventory_in_venv(target.python, target.pkg_name)

    scripts_dir = args.scripts or (args.tasks.parent / SCRIPTS_DIRNAME)
    scripts = load_scripts(scripts_dir)
    tasks = load_tasks(args.tasks)
    coverage = measure_coverage(inventory, scripts)

    if args.queue:
        # Machine-friendly: just the uncovered symbols, one per line, for feeding into a targeted
        # generation run's --feedback.
        for name in coverage.uncovered:
            print(name)
        return 0

    covered_n = len(coverage.covered)
    total_n = len(inventory.names)
    print(f"\nAPI coverage of {inventory.package} {inventory.version}: {covered_n}/{total_n} ({coverage.rate:.0%})")
    print(f"  ground-truth scripts read from {scripts_dir} ({len(scripts)} script(s), {len(tasks)} task(s))")
    missing = [t.id for t in tasks if t.id not in scripts]
    if missing:
        print(f"  {len(missing)} task(s) have no confirmation script (contribute no coverage): {', '.join(missing)}")
    if coverage.uncovered:
        print(f"\nuncovered — the generation queue ({len(coverage.uncovered)} symbols):")
        for name in coverage.uncovered:
            print(f"  {name}")
        print("\nfeed these to a targeted run: `acumen coverage --queue | ...` then `acumen tasks --feedback`")
    else:
        print("\nevery inventory symbol is exercised by at least one task — no generation queue")
    if args.skill:
        _print_skill_coverage(inventory, coverage.covered, args.skills, args.skill, target.pkg_name)
    return 0


def _print_skill_coverage(inventory, covered: frozenset[str], skills_root: Path, version: str, pkg: str) -> None:
    """Benchmark coverage beside skill coverage: what is verified, what is merely taught, what is neither.

    The loop can only optimize what the benchmark scores, so any symbol the skill teaches but no task
    exercises is guidance nothing tests — and the leanness pressure has no reason to keep it. This is
    the map of that gap.
    """
    skill = load_skill(skills_root, version)
    mentioned = skill_mentions(inventory, skill.directory, aliases=[_KNOWN_ALIASES.get(pkg, pkg)])
    names = inventory.names
    verified = covered & mentioned
    taught_only = sorted(mentioned - covered)
    tested_only = sorted(covered - mentioned)
    neither = sorted(names - covered - mentioned)
    print(f"\nskill {version} ({skill.size / 1024:.1f} KB) vs the benchmark, over {len(names)} symbols:")
    print(f"  taught AND verified:      {len(verified)}")
    print(f"  taught, never verified:   {len(taught_only)}  {', '.join(taught_only)[:200]}")
    print(f"  verified, not taught:     {len(tested_only)}  {', '.join(tested_only)[:200]}")
    print(f"  neither:                  {len(neither)}")


def _cmd_lockbox(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    working_path = args.working or args.tasks.with_name(f"{args.tasks.stem}.working.yaml")
    if working_path.exists() and not args.force:
        print(f"{working_path} already exists — pass --force to overwrite it", file=sys.stderr)
        return 2
    working_ids, lock_ids = split_lockbox([t.id for t in tasks], args.fraction, args.seed)
    by_id = {t.id: t for t in tasks}
    try:
        box = write_lockbox(
            args.out, [by_id[i] for i in lock_ids], seed=args.seed, fraction=args.fraction, dump=dump_tasks
        )
    except FoldError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    working_path.write_text(dump_tasks([by_id[i] for i in working_ids]))
    print(f"lockbox: {len(lock_ids)} task(s) held back → {box.directory.resolve()}  ({box.digest[:19]}…)")
    print(f"working: {len(working_ids)} task(s) → {working_path.resolve()}")
    print("\nnext: `acumen loop --cv K --tasks <working> --lockbox <lockbox>`; the lockbox is scored once, at the end")
    return 0


def _print_cv(result: CVResult) -> None:
    """The CV report: per-fold held-out deltas, their mean and spread, then the carried version."""
    print(f"\nrulebook {result.baseline_version} -> {result.carried.version}  (cross-validated, k={len(result.folds)})")
    print(f"  {'fold':<6} {'held-out tasks':<16} {'baseline':<12} {'improved':<12} {'Δ pass':<9} Δ load")
    for f in result.folds:
        b, i = f.baseline_held_out, f.improved_held_out
        print(
            f"  {f.fold.index:<6} {len(f.fold.held_out):<16} {b.passed}/{b.total} ({b.rate:.0%})   "
            f"{i.passed}/{i.total} ({i.rate:.0%})   {f.delta_rate:+.0%}     {f.delta_load_rate:+.0%}"
        )
    print(
        f"  CV estimate: Δ pass {result.cv_mean_delta:+.1%} (spread {result.cv_spread:.1%} across folds); "
        f"Δ load {result.cv_mean_load_delta:+.1%}"
    )
    bt, ct, ns = result.baseline_test, result.carried_test, result.noskill_score
    within = (
        f"skill {result.baseline_skill} {bt.passed}/{bt.total} -> skill {result.carried_skill} {ct.passed}/{ct.total}"
    )
    floor = f"noskill {ns.passed}/{ns.total} -> " if ns.total else ""
    print(f"  within-task test (optimistic, not the estimate): {floor}{within}")
    if result.lockbox is not None:
        print(f"  lockbox: {len(result.lockbox.task_ids)} task(s) untouched — scored only by `acumen loop` at the end")
    else:
        print("  no lockbox: this run cannot support a generalisation claim", file=sys.stderr)
    if not result.carried.changed:
        print("  note: the refit left the rulebook unchanged", file=sys.stderr)
    print(f"  rulebook rationale: {result.carried.rationale}")


def _cmd_loop(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    tasks = load_tasks(args.tasks)
    if args.max_concurrency:
        cfg = replace(cfg, max_concurrency=args.max_concurrency)

    auth_mode = resolve_auth_mode(args.auth)
    _print_auth(auth_mode)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    if cfg.dataset_cache_dirs and not args.no_warm:
        _warm_cache(cfg, target, args.tasks)
    print(
        "PROTOTYPE loop: draft skill from rulebook v1 -> bench -> improve rulebook to v2 -> "
        "draft -> bench. This runs several agents and full benchmark passes; it is slow.",
        flush=True,
    )

    def on_done(outcome: RunOutcome) -> None:
        mark = "✓" if outcome.success else "✗"
        k = outcome.key
        print(f"  {mark} {k.split}/{k.task_id}/rep_{k.rep} ({outcome.reason})", flush=True)

    def on_select(selection: HeadroomSelection) -> None:
        print(f"task selection — headroom on the test split against {', '.join(cfg.models)}:", flush=True)
        print(f"  kept ({len(selection.selected)}): {', '.join(t.id for t in selection.selected) or '—'}")
        if selection.solved:
            print(f"  left out, baseline already solves ({len(selection.solved)}): {', '.join(selection.solved)}")
        if selection.unscreened:
            print(
                f"  left out, never screened ({len(selection.unscreened)}): {', '.join(selection.unscreened)}"
                "  (run `acumen bench --no-skill` on them first)"
            )

    if args.cv:
        stop = StopRule(
            max_iterations=args.iterations,
            patience=args.patience,
            min_delta=args.min_delta,
            max_wallclock_s=args.max_hours * 3600 if args.max_hours else None,
        )
        print(
            f"CV loop: k={args.cv}, up to {stop.max_iterations} iteration(s), patience {stop.patience}"
            + (f", wall-clock cap {args.max_hours:g}h" if args.max_hours else "")
            + ("; NO LOCKBOX" if args.no_lockbox else f"; lockbox {args.lockbox}"),
            flush=True,
        )

        def on_iteration(i: int, cv: CVResult) -> None:
            print(f"\n=== iteration {i} ===")
            _print_cv(cv)
            diff_path = args.rulebooks / cv.carried.version / f"from-{cv.baseline_version}.diff"
            diff_path.write_text(cv.rulebook_diff)
            print(f"  rulebook diff: {diff_path}", flush=True)

        run = asyncio.run(
            run_loop(
                cfg=cfg,
                target=target,
                skills_root=args.skills,
                rulebooks_root=args.rulebooks,
                runs_root=args.runs,
                tasks=tasks,
                k=args.cv,
                seed=args.seed,
                stop=stop,
                lockbox_dir=None if args.no_lockbox else args.lockbox,
                allow_no_lockbox=args.no_lockbox,
                auth_mode=auth_mode,
                task_ids=args.task,
                feedback=args.feedback,
                log_dir=args.log_dir,
                stream=args.stream,
                headroom_only=args.headroom,
                on_select=on_select,
                on_iteration=on_iteration,
                on_fold=lambda f: print(
                    f"  fold {f.fold.index}: held-out {f.improved_held_out.passed}/{f.improved_held_out.total} "
                    f"vs baseline {f.baseline_held_out.passed}/{f.baseline_held_out.total} ({f.delta_rate:+.0%})",
                    flush=True,
                ),
                on_bench_done=on_done,
            )
        )
        print(f"\n=== loop finished: {run.stopped_because} ===")
        print(f"  pick by CV: rulebook {run.best_version} (cross-validated held-out pass rate {run.best_cv_rate:.0%})")
        if run.lockbox_score is not None and run.lockbox_baseline is not None:
            lb, ls = run.lockbox_baseline, run.lockbox_score
            print(
                f"  LOCKBOX (scored once, never selected on): v1 {lb.passed}/{lb.total} ({lb.rate:.0%})  ->  "
                f"{run.best_version} {ls.passed}/{ls.total} ({ls.rate:.0%})   Δ {run.lockbox_delta:+.0%}"
            )
        else:
            print("  no lockbox — no generalisation claim can be made from this run", file=sys.stderr)
        return 0

    result = asyncio.run(
        run_iteration(
            cfg=cfg,
            target=target,
            skills_root=args.skills,
            rulebooks_root=args.rulebooks,
            runs_root=args.runs,
            tasks=tasks,
            auth_mode=auth_mode,
            task_ids=args.task,
            feedback=args.feedback,
            log_dir=args.log_dir,
            stream=args.stream,
            headroom_only=args.headroom,
            on_select=on_select,
            on_bench_done=on_done,
        )
    )

    base, imp, ns = result.baseline_score, result.improved_score, result.noskill_score
    print(f"\nrulebook {result.baseline_version} -> {result.improved_version}")
    if ns.total:
        print(
            f"  held-out PASS: noskill {ns.passed}/{ns.total} ({ns.rate:.0%})  ->  "
            f"skill {result.baseline_skill} {base.passed}/{base.total} ({base.rate:.0%})  ->  "
            f"skill {result.improved_skill} {imp.passed}/{imp.total} ({imp.rate:.0%})"
        )
        # Load rate is the earlier signal: a skill that never loads cannot pass. noskill never loads.
        print(
            f"  held-out LOAD:                       "
            f"skill {result.baseline_skill} {base.loaded}/{base.total} ({base.load_rate:.0%})  ->  "
            f"skill {result.improved_skill} {imp.loaded}/{imp.total} ({imp.load_rate:.0%})"
        )
        print(
            f"  skill vs no-skill: {imp.passed - ns.passed:+d} passes;   "
            f"rulebook v1 vs v2: {result.moved:+d} passes, {result.load_moved:+d} loads"
        )
    else:
        print(
            f"  held-out (test): pass {base.passed}/{base.total} ({base.rate:.0%}) -> "
            f"{imp.passed}/{imp.total} ({imp.rate:.0%}) [{result.moved:+d}];  "
            f"load {base.loaded}/{base.total} ({base.load_rate:.0%}) -> "
            f"{imp.loaded}/{imp.total} ({imp.load_rate:.0%}) [{result.load_moved:+d}]"
        )
    print(
        f"  baseline train pass rate:  {result.baseline_train_score.passed}/"
        f"{result.baseline_train_score.total} ({result.baseline_train_score.rate:.0%})  (drove the improvement)"
    )
    if not result.rulebook.changed:
        print("  note: the improve agent left the rulebook unchanged — no movement is expected", file=sys.stderr)
    print(f"  rulebook rationale: {result.rulebook.rationale}")

    diff_path = args.rulebooks / result.improved_version / f"from-{result.baseline_version}.diff"
    diff_path.write_text(result.rulebook_diff)
    print(f"  rulebook diff: {diff_path}  ({result.rulebook_diff.count(chr(10))} lines)")
    print(f"  cost: ${result.cost_usd:.2f}")
    print(
        "\nNOTE (prototype): held-out here is the within-task test variant, not a task partition; "
        "a single iteration is not protected against selection leakage. This measures whether the "
        "rulebook loop moves a score, not a trustworthy final number."
    )
    return 0


def _cmd_ship(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.model:
        cfg = replace(cfg, ship_model=args.model)

    # Validate the version exists before the (costly) target prep.
    load_skill(args.skills, args.version, expect_name=cfg.skill_name)

    where = (
        "a local path — the change is written to the working tree"
        if cfg.is_local
        else ("a GitHub URL — the change is delivered as a pull request")
    )
    print(f"shipping {args.version} of {cfg.skill_name} into {cfg.repo} ({where})")
    auth_mode = resolve_auth_mode(args.auth)
    _print_auth(auth_mode)
    print(f"preparing target {cfg.repo}@{cfg.ref} ...", flush=True)
    target = prepare_target(cfg, args.cache, refresh=args.refresh_target)
    print(f"target ready: {target.fingerprint} @ {target.commit[:8]}", flush=True)
    print(
        f"running the ship agent with {cfg.ship_model} (real env: it builds, installs, and "
        f"{'opens a PR' if not cfg.is_local else 'edits the working tree'}) ...",
        flush=True,
    )

    log = LiveLog.open(args.log_dir, "ship", stream=args.stream)
    print(f"log → {log.jsonl_path}", flush=True)
    with log:
        result = asyncio.run(
            ship_skill(
                cfg=cfg,
                target=target,
                skills_root=args.skills,
                version=args.version,
                auth_mode=auth_mode,
                max_turns=args.max_turns,
                max_usd=args.max_usd,
                force=args.force,
                log=log,
            )
        )
    print(f"\nshipped {result.skill.version} of {result.skill.name}")
    print(f"  mode:  {'pull request' if result.mode == 'github' else 'working tree (local)'}")
    print(f"  cost:  ${result.cost_usd:.2f} over {result.turns} turns")
    _print_log_result(log)
    if result.summary:
        print("\nagent summary:")
        print(result.summary)
    return 0


def _parse_palette(values: list[str] | None) -> dict[str, str]:
    """Parse ``--palette MODEL=COLOUR`` arguments into a mapping.

    The flag repeats, and one value may carry several comma-separated pairs — neither a
    model id nor a colour spec contains a comma, so the split is unambiguous.
    """
    palette = {}
    for value in values or []:
        for pair in value.split(","):
            if not pair.strip():
                continue
            model, sep, color = pair.partition("=")
            if not sep or not model.strip() or not color.strip():
                raise ReportError(f"--palette expects MODEL=COLOUR, got {pair.strip()!r}")
            palette[model.strip()] = color.strip()
    return palette


def _cmd_report(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks) if args.tasks.exists() else None
    if tasks is None:
        print(f"note: {args.tasks} not found — per-task prompts will be omitted", file=sys.stderr)
    skills_root = args.skills if args.skills.is_dir() else None
    if skills_root is None:
        print(f"note: {args.skills} not found — skill rationale/diff will be omitted", file=sys.stderr)
    report = build_report(args.runs, args.out, tasks, skills_root=skills_root, palette=_parse_palette(args.palette))
    df = report.results
    arms = ", ".join(sorted(df["arm_label"].unique(), key=lambda a: (a != "noskill", a)))
    print(f"aggregated {report.n_runs} runs across arms: {arms}")
    for arm in sorted(df["arm_label"].unique(), key=lambda a: (a != "noskill", a)):
        group = df[df["arm_label"] == arm]
        print(f"  {arm}: {int(group['success'].sum())}/{len(group)} passed")
    print(f"wrote {args.out.resolve()}")
    print(f"wrote {args.out.resolve().with_suffix('.csv')}")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    written = scaffold(args.directory, force=args.force)
    for path in written:
        print(f"wrote {path}")
    print("\nnext: edit config.yaml (repo) and tasks.yaml, then `acumen draft`")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``acumen`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="acumen", description="Build, benchmark, and optimize Claude skills for Python packages."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="run a benchmark pass")
    _add_bench_args(bench)
    _add_auth_arg(bench)
    bench.set_defaults(func=_cmd_bench)

    draft = sub.add_parser("draft", help="draft a skill from the target package's source")
    draft.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    draft.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    draft.add_argument("--model", help="override config draft_model")
    draft.add_argument("--max-turns", type=int, help="cap turns for the drafting agent (default: unbounded)")
    draft.add_argument("--max-usd", type=float, help="cap spend for the drafting agent (default: unbounded)")
    draft.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    draft.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    draft.add_argument("--force", action="store_true", help="draft another version even if some already exist")
    _add_auth_arg(draft)
    _add_feedback_arg(draft, extra=" (e.g. package context, what the skill should emphasise)")
    _add_log_args(draft)
    draft.set_defaults(func=_cmd_draft)

    improve = sub.add_parser("improve", help="improve the current skill into a new version from its train results")
    improve.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    improve.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    improve.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    improve.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    improve.add_argument("--from", dest="from_version", metavar="VERSION", help="version to improve (default: latest)")
    improve.add_argument("--model", help="override config improve_model")
    improve.add_argument("--max-turns", type=int, help="cap turns for the improving agent (default: unbounded)")
    improve.add_argument("--max-usd", type=float, help="cap spend for the improving agent (default: unbounded)")
    improve.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    improve.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    _add_auth_arg(improve)
    _add_feedback_arg(
        improve,
        extra=" (e.g. what to fix or emphasise; do NOT paste test-split answers — that defeats the held-out split)",
    )
    _add_log_args(improve)
    improve.set_defaults(func=_cmd_improve)

    tasks_cmd = sub.add_parser("tasks", help="autonomously generate a tasks.yaml from the target package")
    tasks_cmd.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    tasks_cmd.add_argument("--out", type=Path, default=Path("tasks.yaml"), help="tasks.yaml to write")
    tasks_cmd.add_argument("--model", help="override config tasks_model")
    tasks_cmd.add_argument("--max-turns", type=int, help="cap turns for the generation agent (default: unbounded)")
    tasks_cmd.add_argument("--max-usd", type=float, help="cap spend for the generation agent (default: unbounded)")
    tasks_cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    tasks_cmd.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    tasks_cmd.add_argument("--force", action="store_true", help="overwrite an existing tasks file")
    tasks_cmd.add_argument(
        "--per-notebook",
        action="store_true",
        help="shard generation: run one agent per tutorial notebook, resumable, merged at the end "
        "(scales to exhaustive coverage instead of one agent covering the whole package)",
    )
    tasks_cmd.add_argument(
        "--shards-dir",
        type=Path,
        default=None,
        help="where per-notebook shard files live (default: <out>.shards next to --out); "
        "delete a shard file to regenerate just that notebook",
    )
    tasks_cmd.add_argument(
        "--notebook",
        metavar="SUBSTR",
        action="append",
        help="with --per-notebook: only shard notebooks whose path contains SUBSTR (repeatable)",
    )
    tasks_cmd.add_argument(
        "--candidates",
        type=Path,
        metavar="DIR",
        help="shard over the mined candidate scripts in DIR (from `acumen mine`) instead of notebooks",
    )
    _add_auth_arg(tasks_cmd)
    _add_feedback_arg(tasks_cmd, extra=" (e.g. which functionality to skip or focus on)")
    _add_log_args(tasks_cmd)
    tasks_cmd.set_defaults(func=_cmd_tasks)

    screen_cmd = sub.add_parser(
        "screen",
        help="report baseline (no-skill) pass rate per task as a difficulty signal (run bench --no-skill first)",
    )
    screen_cmd.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    screen_cmd.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    screen_cmd.add_argument("--split", choices=SPLITS, action="append", help="restrict to a split (repeatable)")
    screen_cmd.add_argument(
        "--by-model", action="store_true", help="keep each reference model's runs apart instead of pooling them"
    )
    screen_cmd.set_defaults(func=_cmd_screen)

    coverage_cmd = sub.add_parser(
        "coverage",
        help="report which of the target's public API symbols the task set exercises (the rest are the queue)",
    )
    coverage_cmd.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    coverage_cmd.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    coverage_cmd.add_argument(
        "--scripts", type=Path, help="dir of ground-truth scripts (default: 'scripts/' beside tasks.yaml)"
    )
    coverage_cmd.add_argument("--queue", action="store_true", help="print only the uncovered symbols, one per line")
    coverage_cmd.add_argument(
        "--skill", metavar="VERSION", help="also report what this skill version teaches vs what the benchmark verifies"
    )
    coverage_cmd.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    coverage_cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    coverage_cmd.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    coverage_cmd.set_defaults(func=_cmd_coverage)

    warm_cmd = sub.add_parser(
        "warm",
        help="pre-download the datasets the tasks' ground-truth scripts load into the shared per-target cache",
    )
    warm_cmd.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    warm_cmd.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    warm_cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    warm_cmd.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    warm_cmd.set_defaults(func=_cmd_warm)

    lockbox_cmd = sub.add_parser(
        "lockbox",
        help="hold back a fraction of tasks, written once, never optimized on, scored once at the very end",
    )
    lockbox_cmd.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="the full task set")
    lockbox_cmd.add_argument("--out", type=Path, default=Path("lockbox"), help="lockbox directory to create")
    lockbox_cmd.add_argument(
        "--working", type=Path, help="where to write the remaining (working) tasks (default: <tasks>.working.yaml)"
    )
    lockbox_cmd.add_argument("--fraction", type=float, default=0.2, help="fraction of tasks to hold back (0.2)")
    lockbox_cmd.add_argument("--seed", type=int, default=0, help="selection seed (default 0)")
    lockbox_cmd.add_argument("--force", action="store_true", help="overwrite an existing working file")
    lockbox_cmd.set_defaults(func=_cmd_lockbox)

    mine = sub.add_parser(
        "mine",
        help="harvest real analyses that use the target (GitHub code search, given web pages) into candidate scripts",
    )
    mine.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    mine.add_argument("--out", type=Path, default=Path("mined"), help="directory of candidate scripts (additive)")
    mine.add_argument(
        "--query", action="append", help="a GitHub code-search query (repeatable; default: import + namespaces)"
    )
    mine.add_argument("--alias", help="the package's conventional import alias for default queries (e.g. sq)")
    mine.add_argument("--limit", type=int, default=100, help="max hits per query per file type (default 100)")
    mine.add_argument("--url", action="append", help="a page/notebook/script URL to lift code from (repeatable)")
    mine.add_argument("--no-github", action="store_true", help="skip GitHub; only the given --url pages")
    mine.add_argument(
        "--exclude-repo",
        metavar="OWNER/NAME",
        action="append",
        help="skip hits from this repository (repeatable; the target and its submodules are always skipped)",
    )
    mine.add_argument(
        "--min-symbols", type=int, default=1, help="public API symbols a candidate must reference (default 1)"
    )
    mine.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    mine.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    mine.set_defaults(func=_cmd_mine)

    loop = sub.add_parser(
        "loop",
        help="PROTOTYPE: optimize the rulebook one iteration (draft->bench->improve rulebook->draft->bench)",
    )
    loop.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    loop.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml")
    loop.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    loop.add_argument("--rulebooks", type=Path, default=Path("rulebooks"), help="root of the rulebook tree")
    loop.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    loop.add_argument("--task", metavar="ID", action="append", help="restrict to a task id (repeatable)")
    loop.add_argument("--max-concurrency", type=int, help="override config max_concurrency")
    loop.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    loop.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    loop.add_argument("--no-warm", action="store_true", help="skip pre-downloading datasets into the shared cache")
    loop.add_argument(
        "--headroom",
        action="store_true",
        help="score only tasks the no-skill baseline does not already pass on the test split "
        "(per config model; needs prior `bench --no-skill` runs — unscreened tasks are left out)",
    )
    loop.add_argument(
        "--cv",
        type=int,
        metavar="K",
        help="cross-validate over tasks with K folds: improve on K-1 folds, score on the held-out fold, "
        "report the mean; carry forward the refit on all tasks",
    )
    loop.add_argument("--seed", type=int, default=0, help="fold assignment seed (default 0)")
    loop.add_argument("--iterations", type=int, default=1, help="with --cv: max rulebook iterations (default 1)")
    loop.add_argument(
        "--patience", type=int, default=2, help="with --cv: stop after this many non-improving iterations (2)"
    )
    loop.add_argument(
        "--min-delta", type=float, default=0.0, help="with --cv: CV rate gain that counts as improvement (0)"
    )
    loop.add_argument("--max-hours", type=float, help="with --cv: wall-clock cap, checked between iterations")
    loop.add_argument("--lockbox", type=Path, default=Path("lockbox"), help="lockbox dir from `acumen lockbox`")
    loop.add_argument(
        "--no-lockbox", action="store_true", help="run --cv without a lockbox (no generalisation claim possible)"
    )
    _add_auth_arg(loop)
    _add_feedback_arg(loop)
    _add_log_args(loop)
    loop.set_defaults(func=_cmd_loop)

    ship = sub.add_parser("ship", help="make a benchmarked skill installable into the target package")
    ship.add_argument(
        "--skill", dest="version", metavar="VERSION", required=True, help="skill version to ship, e.g. v2"
    )
    ship.add_argument("--config", type=Path, default=Path("config.yaml"), help="path to config.yaml")
    ship.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree")
    ship.add_argument("--model", help="override config ship_model")
    ship.add_argument("--max-turns", type=int, help="cap turns for the ship agent (default: unbounded)")
    ship.add_argument("--max-usd", type=float, help="cap spend for the ship agent (default: unbounded)")
    ship.add_argument("--cache", type=Path, default=DEFAULT_CACHE_ROOT, help="target cache root")
    ship.add_argument("--refresh-target", action="store_true", help="rebuild the target checkout and venv")
    ship.add_argument("--force", action="store_true", help="ship even if the package already has an installer")
    _add_auth_arg(ship)
    _add_log_args(ship)
    ship.set_defaults(func=_cmd_ship)

    report = sub.add_parser("report", help="aggregate the run tree into a self-contained report.html")
    report.add_argument("--runs", type=Path, default=Path("runs"), help="root of the run tree")
    report.add_argument("--tasks", type=Path, default=Path("tasks.yaml"), help="path to tasks.yaml (for task text)")
    report.add_argument("--skills", type=Path, default=Path("skills"), help="root of the skill tree (rationale/diff)")
    report.add_argument("--out", type=Path, default=Path("report.html"), help="output HTML path (overwritten)")
    report.add_argument(
        "--palette",
        action="append",
        metavar="MODEL=COLOUR",
        help="recolour a model's bars, e.g. --palette claude-opus-5=#3b7ea1 (repeatable, or comma-separated)",
    )
    report.set_defaults(func=_cmd_report)

    init = sub.add_parser("init", help="scaffold a starter config.yaml and tasks.yaml")
    init.add_argument("--dir", type=Path, default=Path("."), dest="directory", help="directory to scaffold into")
    init.add_argument("--force", action="store_true", help="overwrite existing config.yaml / tasks.yaml")
    init.set_defaults(func=_cmd_init)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Returns
    -------
    A process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        ConfigError,
        TaskError,
        EnvError,
        CoverageError,
        MiningError,
        FoldError,
        SkillError,
        DraftError,
        ImproveError,
        TaskGenError,
        ShipError,
        ReportError,
        InitError,
        LoopError,
        RulebookError,
    ) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except TransientLimitError as err:
        print(f"\npaused: {err}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted — completed runs are preserved; rerun to resume", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
