"""The task-generation agent: mine the target package for real analyses and write ``tasks.yaml``.

``acumen tasks`` autonomously benchmarks the target's *functionalities* — the analyses a user
would actually run — and writes a benchmark-ready ``tasks.yaml``. Like the drafter it
reads the package **source** (it has to understand the API to design a real pipeline) and, like
no benchmark agent, it also **runs code in the venv**: every ground-truth answer is obtained by
executing the pipeline and reading the real output, never by copying doc output.

There is no test-split guard here (unlike the improver) — no runs exist yet, so there is nothing
to leak. Isolation is otherwise the same as the other meta-agents: scrubbed env, throwaway
``HOME`` and ``CLAUDE_CONFIG_DIR``.

**Existing skills must not bias task generation.** A target repo may already ship skills or
agent-instruction files (``SKILL.md``, ``.claude/skills/``, ``CLAUDE.md``, ``AGENTS.md``,
``.cursor/``, Copilot instructions). If the generator read them, it would mine the tasks the
author already anticipated and phrase them the way the skill does — defeating the point of an
independent benchmark. So, as with the test-split guard, this is enforced two ways: the agent
reads a **filtered copy** of the source with those artifacts stripped out
(:func:`build_filtered_source`), and a ``PreToolUse`` hook denies any tool call that resolves to
one of them — or to the original unfiltered tree — wherever the agent points it
(:func:`find_skill_access`). ``setting_sources=[]`` additionally means no skill is ever
discovered or loaded into the generator itself.

The script the agent writes to confirm an answer is **scratch**: it lives in a throwaway work
dir that is deleted when this returns, so nothing about the ground-truth pipeline is persisted —
only the task (prompt + answer) lands in ``tasks.yaml``. The output is validated through
:func:`acumen.tasks.parse_tasks` before it is written, so ``acumen tasks`` can never emit a
``tasks.yaml`` the rest of the pipeline would reject.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

from acumen.config import Config
from acumen.env import AuthMode, Target, build_agent_env
from acumen.logs import LiveLog
from acumen.paths import slugify
from acumen.procs import label_env, reap
from acumen.prompts import taskgen_prompt, taskgen_shard_prompt
from acumen.tasks import Task, TaskError, load_tasks, parse_tasks

#: The filename the generation agent writes and we harvest from its work dir.
TASKS_FILE = "tasks.yaml"

#: Basenames anywhere in the tree that are agent-facing skill/guidance artifacts. Reading any
#: of these would let existing guidance bias which tasks the generator mines, so they are both
#: stripped from the source copy the agent reads and denied by the guard hook.
_ARTIFACT_BASENAMES = frozenset({"SKILL.md", "CLAUDE.md", "AGENTS.md", "copilot-instructions.md"})

#: Directory names that hold agent guidance; dropped from the copy and denied wherever they
#: appear on a resolved path (``.claude/skills``, ``.cursor/rules``, …).
_ARTIFACT_DIRS = frozenset({".claude", ".cursor"})

#: tool_input keys carrying a filesystem path — the coarse set the improver's guard also uses.
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename")

#: Shell metacharacters we split a Bash command on to recover path-like tokens.
_SHELL_SPLIT = str.maketrans(dict.fromkeys("\"'`|&;<>()$" + "{}", " "))


class TaskGenError(RuntimeError):
    """Raised when tasks could not be generated."""


@dataclass(frozen=True)
class TaskGenResult:
    """The outcome of a generation run: the tasks written and what it cost."""

    tasks: list[Task]
    out_path: Path
    cost_usd: float
    turns: int
    #: Live log paths for this run, when a :class:`LiveLog` was attached.
    log_jsonl: Path | None = None
    log_html: Path | None = None


# ── Skill-bias isolation ───────────────────────────────────────────────────────────────


def _copy_ignore(root: Path):
    """A ``shutil.copytree`` ignore callback that drops skill/guidance artifacts (and ``.git``).

    Skill directories and agent-instruction files are removed so the generator physically
    cannot read them. A top-level ``skills/`` is treated as agent skills (the packaging
    convention) and dropped at the root only, so a legitimately-named source directory deeper
    in the tree is left alone.
    """
    root = root.resolve()

    def ignore(dir_path: str, names: list[str]) -> set[str]:
        here = Path(dir_path).resolve()
        drop: set[str] = set()
        for name in names:
            if name == ".git" or name in _ARTIFACT_DIRS or name in _ARTIFACT_BASENAMES:
                drop.add(name)
            elif here == root and name == "skills":
                drop.add(name)
        return drop

    return ignore


def build_filtered_source(src: Path, dest: Path) -> Path:
    """Copy ``src`` to ``dest`` with skills and agent-guidance stripped out.

    This is the structural half of the skill-bias isolation: the generator's ``add_dirs`` points
    at the returned copy, not the real checkout, so existing skills are simply absent from what
    it can read.

    Parameters
    ----------
    src
        The real target source checkout.
    dest
        Where to write the filtered copy; must not already exist.

    Returns
    -------
    ``dest``.
    """
    shutil.copytree(src, dest, ignore=_copy_ignore(src), symlinks=True)
    return dest


def _artifact_hit(candidate: str, original_src: Path) -> str | None:
    """Return ``candidate`` if it resolves to a skill/guidance artifact or the original tree."""
    try:
        resolved = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if resolved.name in _ARTIFACT_BASENAMES:
        return candidate
    if set(resolved.parts) & _ARTIFACT_DIRS:
        return candidate
    # The unfiltered source tree is off-limits — the agent must read the filtered copy, so any
    # path back into the original checkout (which still holds the stripped artifacts) is denied.
    try:
        resolved.relative_to(original_src)
    except ValueError:
        return None
    return candidate


def find_skill_access(tool_name: str, tool_input: dict[str, Any], original_src: Path) -> str | None:
    """Return the first path in a tool call that reaches a skill/guidance artifact, else ``None``.

    Pure and side-effect free, so the enforcement can be exercised directly without standing up
    an agent (mirrors :func:`acumen.improve.find_test_access`). Checks the path-bearing
    tool_input keys and — for shell tools — the metacharacter-split command tokens, since a Bash
    call can name a path no structured field would.

    Parameters
    ----------
    tool_name
        The tool being invoked; unused today but kept so the guard can special-case tools.
    tool_input
        The tool's arguments.
    original_src
        The real (unfiltered) source checkout, resolved by the caller.

    Returns
    -------
    The offending path string, or ``None`` if the call touches no artifact.
    """
    for key in _PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            hit = _artifact_hit(value, original_src)
            if hit is not None:
                return hit
    command = tool_input.get("command")
    if isinstance(command, str):
        for raw in command.translate(_SHELL_SPLIT).split():
            token = raw.rstrip(",;")
            if not token:
                continue
            hit = _artifact_hit(token, original_src)
            if hit is not None:
                return hit
    return None


def make_skill_guard(original_src: Path) -> HookMatcher:
    """Build the ``PreToolUse`` hook that denies the generator any existing skill/guidance.

    ``matcher=None`` fires the hook for every tool. Paths are resolved against the real source
    checkout, so the guard holds regardless of the agent's ``cwd``.
    """
    root = original_src.resolve()

    async def guard(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        hit = find_skill_access(
            input_data.get("tool_name", ""),
            input_data.get("tool_input", {}) or {},
            root,
        )
        if hit is None:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"acumen hides existing skills and agent-instruction files from the task "
                    f"generator so they cannot bias task selection ({hit}). Design tasks from the "
                    "package's API and its user-facing docs only, using the provided source copy."
                ),
            }
        }

    return HookMatcher(matcher=None, hooks=[guard])


# ── Task serialisation ─────────────────────────────────────────────────────────────────


def _task_to_dict(task: Task) -> dict[str, object]:
    """Serialise a :class:`Task` back to the ``tasks.yaml`` mapping shape."""
    entry: dict[str, object] = {
        "id": task.id,
        "train": {"prompt": task.train.prompt, "answer": task.train.answer},
        "test": {"prompt": task.test.prompt, "answer": task.test.answer},
    }
    if task.max_turns is not None:
        entry["max_turns"] = task.max_turns
    if task.max_usd is not None:
        entry["max_usd"] = task.max_usd
    if task.model is not None:
        entry["model"] = task.model
    return entry


class _TaskDumper(yaml.SafeDumper):
    """A SafeDumper that renders multi-line strings as literal blocks, for readable prompts."""


def _represent_str(dumper: yaml.Dumper, value: str) -> yaml.ScalarNode:
    # Multi-line prompts are far easier to review as `|` literal blocks than folded/quoted.
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_TaskDumper.add_representer(str, _represent_str)


def dump_tasks(tasks: list[Task]) -> str:
    """Render tasks as ``tasks.yaml`` text, preserving key order and multi-line prompts.

    The inverse of :func:`acumen.tasks.load_tasks`. Round-tripping through this is how the
    combined (existing + generated) task set is validated before anything is written to disk.
    Multi-line prompts are emitted as ``|`` literal blocks so a reviewer can read them.
    """
    doc = {"tasks": [_task_to_dict(task) for task in tasks]}
    return yaml.dump(doc, Dumper=_TaskDumper, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100)


def _validate_generated(staged: Path) -> list[Task]:
    """Load and validate the agent's ``tasks.yaml``, mapping failures to a TaskGenError."""
    if not staged.is_file():
        raise TaskGenError(
            f"the task-generation agent did not write {TASKS_FILE} — nothing to save. "
            "Inspect the prompt or raise max_turns."
        )
    try:
        return load_tasks(staged)
    except TaskError as err:
        raise TaskGenError(f"the generated {TASKS_FILE} is not valid: {err}") from err


async def generate_tasks(
    *,
    cfg: Config,
    target: Target,
    out_path: Path,
    auth_mode: AuthMode = "session",
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    force: bool = False,
    feedback: str | None = None,
    log: LiveLog | None = None,
) -> TaskGenResult:
    """Generate a ``tasks.yaml`` for the target package by mining and executing its analyses.

    Parameters
    ----------
    cfg
        The pass config; supplies ``tasks_model``.
    target
        The prepared target, supplying the source checkout and the interpreter to run
        pipelines against for ground truth.
    auth_mode
        Which credential the generation agent authenticates with — ``"session"`` (the Claude
        subscription) or ``"api"`` (see :func:`acumen.env.build_agent_env`).
    out_path
        Where the tasks are written. Refuses to overwrite an existing file unless ``force``
        is set. There is deliberately no "append" mode: the agent generates blind to the
        existing file (it never reads it — see the module docstring on isolation), so it cannot
        avoid re-covering functionality already present, and appending would silently grow the
        set with semantic duplicates. To combine generated tasks with a curated file, write to a
        separate ``out_path`` and merge by hand, where the overlap can actually be judged.
    model
        Override for the generation model; defaults to ``cfg.tasks_model``.
    max_turns, max_usd
        Caps for the generation agent. **Unbounded by default**: generating tasks means
        running package code iteratively, so no default budget is imposed — pass explicit caps
        to bound it.
    force
        Overwrite an existing ``out_path`` (e.g. the placeholder from ``acumen init``).
    feedback
        Optional maintainer guidance, injected into the generation prompt as a subordinated
        block — e.g. which functionality to skip. Nothing but ``tasks.yaml`` is persisted (task
        generation writes no meta), so the feedback is not recorded on disk.
    log
        A :class:`LiveLog` to stream the agent's messages to and render an HTML log from.

    Returns
    -------
    The generation result: the task set written and what it cost.
    """
    if out_path.exists() and not force:
        raise TaskGenError(f"{out_path} already exists — pass force=True to overwrite it")

    holder = Path(tempfile.mkdtemp(prefix="acumen-tasks-"))
    try:
        # The agent reads a copy of the source with skills/agent-guidance stripped, never the
        # real checkout — so existing skills cannot bias the tasks it generates.
        source_copy = build_filtered_source(target.src_dir, holder / "source")
        generated, result = await _run_generation_agent(
            work_root=holder,
            source_copy=source_copy,
            target=target,
            make_prompt=lambda staged: taskgen_prompt(
                package=target.pkg_name,
                src=source_copy,
                python=target.python,
                out=staged,
                feedback=feedback,
            ),
            model=model or cfg.tasks_model,
            max_turns=max_turns,
            max_usd=max_usd,
            auth_mode=auth_mode,
            extra_allow=cfg.env_passthrough,
            log=log,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dump_tasks(generated))
        return TaskGenResult(
            tasks=generated,
            out_path=out_path,
            cost_usd=result.total_cost_usd or 0.0,
            turns=result.num_turns,
            log_jsonl=log.jsonl_path if log is not None else None,
            log_html=log.html_path if log is not None and log.html_rendered else None,
        )
    finally:
        # Kill anything the agent left running before removing the directory it runs in.
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)


async def _run_generation_agent(
    *,
    work_root: Path,
    source_copy: Path,
    target: Target,
    make_prompt: Callable[[Path], str],
    model: str,
    max_turns: int | None,
    max_usd: float | None,
    auth_mode: AuthMode,
    extra_allow: Sequence[str],
    log: LiveLog | None,
) -> tuple[list[Task], ResultMessage]:
    """Run one generation agent in ``work_root`` and return its validated tasks and result.

    The isolation-critical core shared by the whole-package generator (:func:`generate_tasks`) and
    every per-notebook one (:func:`generate_tasks_sharded`): a throwaway ``HOME`` and config dir
    under ``work_root``, the scrubbed env, the skill-bias guard resolved against ``target.src_dir``,
    ``add_dirs`` pointed at the (shared, read-only) filtered ``source_copy`` rather than the real
    checkout, and the query loop that streams to ``log``. Keeping this in one place is deliberate:
    it is the security-sensitive part (env scrub + skill guard), and two copies could drift.

    The caller owns ``work_root`` and its teardown (``reap`` + ``rmtree``) — the agent's scratch
    ``script.py`` lives under it and is discarded, so nothing about the ground-truth pipeline is
    persisted, only the tasks it stages. ``make_prompt`` receives the staged ``tasks.yaml`` path
    and returns the prompt, so the whole-package and per-notebook callers differ only in the prompt
    they build over the same isolation.

    Raises
    ------
    TaskGenError
        If the agent failed, produced no result, errored, or wrote no valid ``tasks.yaml``.
    """
    work = work_root / "work"
    home = work_root / "home"
    config_dir = home / ".claude"
    for path in (work, home, config_dir, home / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    staged = work / TASKS_FILE

    # Marks the agent's processes so the caller's teardown can find what it leaves running.
    env = label_env(
        build_agent_env(
            config_dir=config_dir,
            home=home,
            extra_path=[target.bin_dir],
            auth_mode=auth_mode,
            extra_allow=extra_allow,
        ),
        work_root,
    )
    options = ClaudeAgentOptions(
        cwd=str(work),
        env=env,
        model=model,
        # No default budget cap: only bound the agent if the caller asked.
        max_turns=max_turns,
        max_budget_usd=max_usd,
        # The generator reads the target source, like the drafter; benchmark agents never do.
        # It points at the *filtered* copy, not the real checkout.
        add_dirs=[str(source_copy)],
        # No skill discovery at all — the generator must not load a skill that would bias it.
        setting_sources=[],
        permission_mode="bypassPermissions",
        system_prompt={"type": "preset", "preset": "claude_code"},
        # Belt-and-braces over the filtered copy: deny any call that reaches an existing
        # skill/guidance artifact or the original unfiltered source, wherever pointed.
        hooks={"PreToolUse": [make_skill_guard(target.src_dir)]},
    )

    result: ResultMessage | None = None
    agent_error: Exception | None = None
    try:
        async for message in query(prompt=make_prompt(staged), options=options):
            if log is not None:
                log.append(message)
            if isinstance(message, ResultMessage):
                result = message
    except Exception as err:  # noqa: BLE001 - a failed generation is an error to report, re-raised below
        agent_error = err
    finally:
        # Render the HTML log while the throwaway config dir still holds the native transcript —
        # in a finally so an aborted run (the SDK raises on a cap breach, after yielding the
        # result) is still inspectable.
        if log is not None:
            log.finalize(config_dir=config_dir, work_dir=work, result=result)

    if agent_error is not None:
        raise TaskGenError(
            f"the task-generation agent failed: {type(agent_error).__name__}: {agent_error}"
        ) from agent_error
    if result is None:
        raise TaskGenError("the task-generation agent produced no result message")
    if result.is_error:
        raise TaskGenError(f"the task-generation agent errored: {result.subtype} {result.errors or ''}".strip())

    return _validate_generated(staged), result


# ── Sharded generation (one agent per notebook) ────────────────────────────────────────


#: Directories under a checkout that hold ``.ipynb`` files that are not tutorials.
_NOTEBOOK_SKIP_DIRS = frozenset({".ipynb_checkpoints"})


def notebook_shards(source: Path) -> list[Path]:
    """Return every tutorial notebook under ``source``, as paths relative to it, sorted.

    The shard list for :func:`generate_tasks_sharded`: one notebook is one shard. Enumeration
    happens here in the harness — not inside an agent, as the whole-package generator does — so the
    fan-out, the per-shard resume, and the merge are all driven from a list the code controls rather
    than one an agent rediscovers each run. Jupyter checkpoint copies are skipped: they are stale
    duplicates, not tutorials.
    """
    return [
        nb.relative_to(source) for nb in sorted(source.rglob("*.ipynb")) if not (set(nb.parts) & _NOTEBOOK_SKIP_DIRS)
    ]


def _shard_slug(notebook: Path) -> str:
    """A filesystem-safe, collision-resistant slug for a notebook shard.

    Built from the notebook's path *relative to the source root*, not just its basename, so two
    notebooks named the same in different galleries (``tutorials/plotting.ipynb`` and
    ``examples/plotting.ipynb``) get distinct shards, distinct shard files, and distinct task-id
    prefixes. This is the shard file's stem and the task-id namespace, so it must round-trip.
    """
    return slugify("-".join(notebook.with_suffix("").parts))


def _namespace_task(task: Task, slug: str) -> Task:
    """Prefix a task's id with its shard slug, so merged ids are unique and traceable.

    Each shard's agent picks ids blind to the other shards, so two shards can independently pick the
    same id. Prefixing with the shard slug (unique per notebook) makes the merged set satisfy
    :func:`acumen.tasks.parse_tasks`' uniqueness rule and lets a reader see which notebook a task
    came from. Both parts are already filesystem-safe, so the result is too; the merge re-validates
    regardless.
    """
    return replace(task, id=f"{slug}__{task.id}")


@dataclass(frozen=True)
class ShardOutcome:
    """The outcome of one notebook shard within a sharded generation pass."""

    #: Notebook path relative to the source root.
    notebook: Path
    slug: str
    shard_path: Path
    #: ``"generated"`` (an agent ran and wrote it now), ``"cached"`` (a valid shard file was already
    #: present and reused), or ``"failed"`` (the agent errored or wrote nothing valid).
    status: str
    n_tasks: int = 0
    cost_usd: float = 0.0
    turns: int = 0
    error: str | None = None
    log_jsonl: Path | None = None
    log_html: Path | None = None

    @property
    def ok(self) -> bool:
        """Whether this shard contributed tasks (freshly generated or reused from cache)."""
        return self.status in ("generated", "cached")


@dataclass(frozen=True)
class ShardedResult:
    """The outcome of a sharded generation pass: the merged task set plus per-shard detail."""

    tasks: list[Task]
    out_path: Path
    shards_dir: Path
    outcomes: list[ShardOutcome]
    cost_usd: float

    @property
    def n_ok(self) -> int:
        """How many shards contributed tasks."""
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def n_failed(self) -> int:
        """How many shards failed."""
        return sum(1 for outcome in self.outcomes if outcome.status == "failed")


def merge_shards(shards_dir: Path, out_path: Path) -> list[Task]:
    """Merge every ``<slug>.yaml`` shard in ``shards_dir`` into one validated ``tasks.yaml``.

    Each shard's task ids are prefixed with the shard slug (the file's stem) so the union has
    unique, traceable ids, then the whole set is round-tripped through the strict loader — so a
    merged file the rest of the pipeline would reject can never be written. Cheap and idempotent:
    re-runnable after any subset of shards regenerates, which is what makes the whole pass resumable.

    Raises
    ------
    TaskGenError
        If no shard files hold any task.
    """
    merged: list[Task] = []
    for shard_file in sorted(shards_dir.glob("*.yaml")):
        for task in load_tasks(shard_file):
            merged.append(_namespace_task(task, shard_file.stem))
    if not merged:
        raise TaskGenError(f"no tasks found across shards in {shards_dir} — nothing to merge")
    # Round-trip through dump + strict parse to enforce cross-shard uniqueness before writing.
    tasks = parse_tasks(yaml.safe_load(dump_tasks(merged)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_tasks(tasks))
    return tasks


def _cached_shard(notebook: Path, slug: str, shard_path: Path) -> ShardOutcome | None:
    """Return a ``"cached"`` outcome if ``shard_path`` already holds a valid shard, else ``None``.

    Resume is by file presence, like the benchmark runner's ``result.json`` check: a shard whose
    file parses is done and its agent is not re-run. A file that is present but *unparseable* is
    treated as absent (returns ``None``) so a truncated write from an interrupted run is regenerated
    rather than trusted.
    """
    if not shard_path.is_file():
        return None
    try:
        tasks = load_tasks(shard_path)
    except TaskError:
        return None
    return ShardOutcome(notebook=notebook, slug=slug, shard_path=shard_path, status="cached", n_tasks=len(tasks))


async def generate_tasks_sharded(
    *,
    cfg: Config,
    target: Target,
    out_path: Path,
    shards_dir: Path,
    auth_mode: AuthMode = "session",
    model: str | None = None,
    max_turns: int | None = None,
    max_usd: float | None = None,
    max_concurrency: int | None = None,
    force: bool = False,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    on_shard_start: Callable[[ShardOutcome], None] | None = None,
    on_shard_done: Callable[[ShardOutcome], None] | None = None,
) -> ShardedResult:
    """Generate ``tasks.yaml`` by fanning out one generation agent per notebook.

    The scalable form of :func:`generate_tasks`. Instead of one agent enumerating and covering the
    whole package in a single context — which serializes the package into one agent's stamina and
    loses everything on a single error — this enumerates the target's notebooks in the harness
    (:func:`notebook_shards`) and runs one :func:`taskgen_shard_prompt` agent per notebook,
    bounded by a semaphore. Each agent writes its own validated ``shards_dir/<slug>.yaml``; a shard
    that fails is recorded and skipped rather than taking the pass down, and a shard whose file
    already parses is reused (resume by file presence). A final :func:`merge_shards` namespaces the
    ids by shard and writes the combined ``out_path``.

    All shards share ONE read-only filtered source copy (built once), so isolation costs a single
    ``copytree`` rather than one per notebook; each shard still gets its own throwaway work/home and
    its own labelled processes, reaped independently.

    Parameters
    ----------
    cfg
        The pass config; supplies ``tasks_model``, ``max_concurrency``, and ``env_passthrough``.
    target
        The prepared target: the source checkout to shard and the venv to run pipelines against.
    out_path
        The merged ``tasks.yaml``. Refuses to overwrite an existing file unless ``force``.
    shards_dir
        Directory of per-shard ``<slug>.yaml`` files. Resume reads it; regenerate a shard by
        deleting its file (``force`` governs only ``out_path``, never the shard cache — deleting is
        the deliberate, inspectable way to redo work).
    auth_mode
        Which credential each agent authenticates with (see :func:`acumen.env.build_agent_env`).
    model
        Override for the generation model; defaults to ``cfg.tasks_model``.
    max_turns, max_usd
        Per-shard caps for each agent. Unbounded by default, as with :func:`generate_tasks`.
    max_concurrency
        Ceiling on simultaneous shard agents; defaults to ``cfg.max_concurrency``.
    force
        Overwrite an existing ``out_path``. Does not clear the shard cache.
    feedback
        Optional maintainer guidance, passed to every shard agent.
    log_dir
        If given, each shard streams a :class:`LiveLog` to ``log_dir/acumen-tasks-<slug>-*.jsonl``.
    stream
        Mirror each shard's log to the terminal (noisy under concurrency; off by default).
    on_shard_start, on_shard_done
        Optional progress callbacks, invoked as each shard is admitted and as it lands.

    Returns
    -------
    The sharded result: the merged tasks, the per-shard outcomes, and the total cost.

    Raises
    ------
    TaskGenError
        If ``out_path`` exists without ``force``, the target has no notebooks, or no shard produced
        any task.
    """
    if out_path.exists() and not force:
        raise TaskGenError(f"{out_path} already exists — pass force=True to overwrite it")

    holder = Path(tempfile.mkdtemp(prefix="acumen-shards-"))
    try:
        # One read-only filtered copy shared by every shard — the isolation, built once.
        source_copy = build_filtered_source(target.src_dir, holder / "source")
        notebooks = notebook_shards(source_copy)
        if not notebooks:
            raise TaskGenError(
                f"no notebooks (*.ipynb) found under {target.src_dir} — sharded generation needs "
                "tutorials to shard on. Check submodules are present, or use generate_tasks to "
                "infer analyses from source instead."
            )
        shards_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrency or cfg.max_concurrency)
        outcomes: list[ShardOutcome] = []

        async def run_shard(notebook: Path) -> ShardOutcome:
            slug = _shard_slug(notebook)
            shard_path = shards_dir / f"{slug}.yaml"
            cached = _cached_shard(notebook, slug, shard_path)
            if cached is not None:
                if on_shard_start is not None:
                    on_shard_start(cached)
                if on_shard_done is not None:
                    on_shard_done(cached)
                return cached

            async with semaphore:
                pending = ShardOutcome(notebook=notebook, slug=slug, shard_path=shard_path, status="failed")
                if on_shard_start is not None:
                    on_shard_start(pending)
                work_root = holder / "shards" / slug
                work_root.mkdir(parents=True, exist_ok=True)
                log = LiveLog.open(log_dir, f"tasks-{slug}", stream=stream) if log_dir is not None else None
                try:
                    tasks, result = await _run_generation_agent(
                        work_root=work_root,
                        source_copy=source_copy,
                        target=target,
                        make_prompt=lambda staged, nb=notebook: taskgen_shard_prompt(
                            package=target.pkg_name,
                            src=source_copy,
                            python=target.python,
                            out=staged,
                            notebook=nb.as_posix(),
                            feedback=feedback,
                        ),
                        model=model or cfg.tasks_model,
                        max_turns=max_turns,
                        max_usd=max_usd,
                        auth_mode=auth_mode,
                        extra_allow=cfg.env_passthrough,
                        log=log,
                    )
                    # Only a fully validated shard is written, so a present file always parses.
                    shard_path.write_text(dump_tasks(tasks))
                    outcome = ShardOutcome(
                        notebook=notebook,
                        slug=slug,
                        shard_path=shard_path,
                        status="generated",
                        n_tasks=len(tasks),
                        cost_usd=result.total_cost_usd or 0.0,
                        turns=result.num_turns,
                        log_jsonl=log.jsonl_path if log is not None else None,
                        log_html=log.html_path if log is not None and log.html_rendered else None,
                    )
                except Exception as err:  # noqa: BLE001 - one shard failing must not take the pass down
                    outcome = ShardOutcome(
                        notebook=notebook,
                        slug=slug,
                        shard_path=shard_path,
                        status="failed",
                        error=f"{type(err).__name__}: {err}",
                        log_jsonl=log.jsonl_path if log is not None else None,
                        log_html=log.html_path if log is not None and log.html_rendered else None,
                    )
                finally:
                    if log is not None:
                        log.close()
                    # Reap this shard's labelled processes and drop its work dir; the shard file it
                    # wrote lives in shards_dir, outside the holder, so it survives.
                    reap(work_root)
                    shutil.rmtree(work_root, ignore_errors=True)
                if on_shard_done is not None:
                    on_shard_done(outcome)
                return outcome

        for coro in asyncio.as_completed([run_shard(nb) for nb in notebooks]):
            outcomes.append(await coro)

        # Merge whatever landed (freshly generated + reused). Fails loudly if every shard failed.
        outcomes.sort(key=lambda o: o.slug)
        tasks = merge_shards(shards_dir, out_path)
        return ShardedResult(
            tasks=tasks,
            out_path=out_path,
            shards_dir=shards_dir,
            outcomes=outcomes,
            cost_usd=sum(outcome.cost_usd for outcome in outcomes),
        )
    finally:
        reap(holder)
        shutil.rmtree(holder, ignore_errors=True)
