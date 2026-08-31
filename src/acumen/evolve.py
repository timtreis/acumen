"""Generational rulebook evolution: improve-from-best, cheap screens, a confirmed ratchet.

The CV loop (:mod:`acumen.loop`) measures one improvement step honestly but costs ~a session window
per iteration, and both live rounds showed its chain walking *through* regressions (iteration 2
improved from a version that was already worse out of sample). This module trades per-decision
certainty for decision volume — the shape of an autoresearch run with hundreds of generations:

* **Improve from the best, always.** Every generation's improve agent starts from the current
  champion and its failure evidence, never from a regressed child. Rejected candidates keep their
  version directories (provenance) but nothing descends from them.
* **Wild exploration.** Each generation carries a rotating *directive* (:data:`DIRECTIVES`) injected
  as maintainer feedback — bold restructures, aggressive deletion, inversion of an assumption —
  so the search tries large moves, not only one-more-bullet edits.
* **Two-tier selection.** A generation is scored on a small rotating **screen** subset (cheap,
  noisy, allowed to be wrong); a screen win only makes the candidate the *screen champion*. The
  **confirmed champion** — the version improvement builds from permanently — only changes after a
  full held-out bench (every ``confirm_every`` accepts), and a failed confirmation reverts the
  screen champion. Screen decisions may thrash; the ratchet may not.
* **Archive everything.** Every generation appends one JSON line (directive, versions, subset,
  scores, decision) to ``runs/evolve.jsonl`` — hundreds of recorded accept/reject decisions are the
  dataset a cross-pollination step mines for edits that replicate across independent islands.
* The **lockbox** stays write-once and is opened once at the very end (champion and seed, each over
  ``n_drafts`` independent drafts) — screens and confirmations never touch it.

Everything resumes by file presence and *determinism*: the directive, screen subset, and candidate
version of generation ``g`` are pure functions of ``(g, seed)``, so a resumed run replays every
decision from on-disk scores without spawning an agent for finished work.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from acumen import rulebooks as rb
from acumen.bench import PlannedRun
from acumen.config import Config
from acumen.difficulty import HeadroomSelection, screen, select_headroom
from acumen.env import AuthMode, Target
from acumen.folds import FoldError, Lockbox, check_disjoint, load_lockbox_tasks, read_lockbox
from acumen.loop import (
    DRAFTS_DIRNAME,
    LOCKBOX_RUNS_DIRNAME,
    DraftScores,
    LoopError,
    Score,
    _bench,
    _ensure_rulebook,
    _ensure_skill,
    _lockbox_eval,
    score,
)
from acumen.paths import SPLITS
from acumen.runner import RunOutcome
from acumen.skills import Skill
from acumen.tasks import Task

#: The journal of decisions, one JSON object per finished generation, under the runs root.
JOURNAL_FILE = "evolve.jsonl"

#: Exploration directives, rotated deterministically over generations (``g-1 mod len``). Each is
#: injected into the improve agent's prompt as maintainer feedback, so the meta.json of every
#: version records which directive shaped it. The mix is deliberate: focused repair, bold
#: restructuring, deletion pressure, concretization, loading/routing work, assumption-inversion,
#: generalization, and a free slot — a search that only ever adds bullets converges on bloat.
DIRECTIVES: tuple[str, ...] = (
    "Make one focused edit addressing the largest cluster of failures in the evidence.",
    "Be bold: restructure the rulebook wholesale if a different organization would generate better "
    "skills — reorder, merge, or split sections freely. Large rewrites are allowed.",
    "Delete aggressively: find the guidance you believe pays for itself least and remove it. "
    "Leanness is a virtue; instruction no failure ever needed is suspect.",
    "Concretize: for the failure classes in the evidence, make the rulebook demand exact, runnable "
    "recipes (function sequence, parameter values) instead of prose advice.",
    "Attack loading and routing: rewrite how the rulebook instructs the skill's description and "
    "SKILL.md-to-references routing, so the right guidance is in front of the agent at the right time.",
    "Question one assumption the rulebook takes for granted and invert or drop it; say in the "
    "rationale which assumption and why the evidence licenses doubting it.",
    "Generalize: find two or more rules that are special cases of one deeper principle and replace "
    "them with that principle.",
    "Free choice: make the change you most believe in, however unconventional.",
)


def directive_for(generation: int) -> str:
    """The exploration directive of 1-based ``generation`` — deterministic, so resume replays it."""
    if generation < 1:
        raise LoopError(f"generation must be >= 1, got {generation}")
    return DIRECTIVES[(generation - 1) % len(DIRECTIVES)]


def screen_subset(ids: Sequence[str], size: int, epoch: int, seed: int) -> list[str]:
    """The screen tasks of ``epoch`` — a seeded sample, stable within an epoch, rotated across them.

    Rotation is the guard against a search that optimizes the screen instead of the package: over
    epochs the subsets cover the whole pool, and a champion is re-screened on every new subset it
    survives into. Deterministic in ``(seed, epoch)`` so a resumed run rebuilds the same subsets.
    """
    pool = sorted(set(ids))
    if size < 1:
        raise LoopError(f"screen size must be >= 1, got {size}")
    if len(pool) <= size:
        return pool
    return sorted(random.Random(f"{seed}:{epoch}").sample(pool, size))


@dataclass(frozen=True)
class Generation:
    """One recorded generation: what was tried, on which screen, and what was decided."""

    index: int
    directive: str
    #: The screen champion the candidate was improved from.
    parent: str
    #: The rulebook/skill version this generation wrote (``v{index+1}``), kept even if rejected.
    candidate: str
    subset: tuple[str, ...]
    champion_screen: Score
    candidate_screen: Score
    accepted: bool
    #: Whether a full-bench confirmation ran after this generation, and what it decided.
    confirm_ran: bool
    #: With ``confirm_ran``: ``True`` promoted the challenger, ``False`` reverted to the incumbent.
    confirm_promoted: bool | None
    #: The confirmed champion after this generation — the version improvement ratchets from.
    confirmed: str
    cost_usd: float

    def to_json(self) -> dict:
        """The journal line — flat, so the archive is greppable and mineable."""
        return {
            "generation": self.index,
            "directive": self.directive,
            "parent": self.parent,
            "candidate": self.candidate,
            "subset": list(self.subset),
            "champion_screen": [self.champion_screen.passed, self.champion_screen.total],
            "candidate_screen": [self.candidate_screen.passed, self.candidate_screen.total],
            "accepted": self.accepted,
            "confirm_ran": self.confirm_ran,
            "confirm_promoted": self.confirm_promoted,
            "confirmed": self.confirmed,
            "cost_usd": round(self.cost_usd, 4),
        }


@dataclass(frozen=True)
class EvolveRun:
    """The outcome of one evolution run: the decisions, the champion, and (optionally) the verdict."""

    generations: list[Generation]
    #: The confirmed champion at the end — pending screen accepts are confirmed (or reverted) first.
    champion: str
    stopped_because: str
    journal_path: Path
    #: Lockbox verdicts over ``n_drafts`` drafts per version; ``None`` when the lockbox was not
    #: opened (islands never open it — only the final merged champion does).
    lockbox_drafts: DraftScores | None = None
    lockbox_baseline_drafts: DraftScores | None = None
    selection: HeadroomSelection | None = None
    cost_usd: float = 0.0

    @property
    def accepted(self) -> int:
        """How many generations won their screen."""
        return sum(1 for g in self.generations if g.accepted)

    @property
    def lockbox_mean_delta(self) -> float | None:
        """Lockbox mean-rate change from the seed to the champion, over all drafts."""
        if self.lockbox_drafts is None or self.lockbox_baseline_drafts is None:
            return None
        return self.lockbox_drafts.mean_rate - self.lockbox_baseline_drafts.mean_rate


def _read_journal(path: Path) -> set[int]:
    """The generation indices already journaled — appends are idempotent across resumes."""
    if not path.is_file():
        return set()
    done: set[int] = set()
    for line in path.read_text().splitlines():
        try:
            done.add(int(json.loads(line)["generation"]))
        except (ValueError, KeyError, TypeError):
            continue
    return done


async def run_evolve(
    *,
    cfg: Config,
    target: Target,
    skills_root: Path,
    rulebooks_root: Path,
    runs_root: Path,
    tasks: Sequence[Task],
    generations: int = 20,
    screen_size: int = 12,
    epoch_len: int = 5,
    accept_delta: int = 2,
    confirm_every: int = 5,
    n_drafts: int = 3,
    seed: int = 0,
    lockbox_dir: Path | None = None,
    allow_no_lockbox: bool = False,
    evaluate_lockbox: bool = True,
    auth_mode: AuthMode = "session",
    max_concurrency: int | None = None,
    feedback: str | None = None,
    log_dir: Path | None = None,
    stream: bool = False,
    headroom_only: bool = False,
    max_wallclock_s: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_select: Callable[[HeadroomSelection], None] | None = None,
    on_generation: Callable[[Generation], None] | None = None,
    on_bench_start: Callable[[PlannedRun], None] | None = None,
    on_bench_done: Callable[[RunOutcome], None] | None = None,
) -> EvolveRun:
    """Run ``generations`` improve-from-best steps with screen selection and a confirmed ratchet.

    Per generation ``g`` (all deterministic in ``(g, seed)``, so resume replays from disk):

    1. Bench the screen champion on this epoch's subset, train+test (train = the improve evidence,
       test = the score the candidate must beat; both mostly cached within an epoch).
    2. Improve the rulebook **from the screen champion** under :func:`directive_for`'s directive
       (composed with the caller's ``feedback``), into version ``v{g+1}``; draft one skill from it.
    3. Bench the candidate on the subset's test split; **accept** (candidate becomes screen
       champion) iff it beats the champion by ``accept_delta`` passes on the identical subset.
    4. Every ``confirm_every`` accepts, bench the screen champion on the FULL task set's test split
       and compare with the confirmed champion's full bench: strictly more passes promotes it;
       anything else reverts the screen champion to the confirmed one. Pending accepts are
       confirmed the same way once the generation budget or wall-clock stops the run.

    ``accept_delta`` exists because the screen is noisy by construction (12-ish tasks, one draft):
    it should be set from measured draft variance, not hope — a screen win smaller than the noise
    floor must not even challenge.

    With ``evaluate_lockbox`` (and a lockbox): the seed and the final champion are scored on the
    lockbox over ``n_drafts`` independent drafts each — the run's one honest number. Pass
    ``evaluate_lockbox=False`` for island runs whose champion feeds cross-pollination instead; the
    lockbox is still read and checked disjoint, and its trees denied to every improve agent.

    Raises
    ------
    LoopError
        On a missing/overlapping lockbox (unless waived), degenerate parameters, or a broken
        version chain (the workspace was tampered with between resumes).
    """
    if generations < 1:
        raise LoopError(f"generations must be >= 1, got {generations}")
    if accept_delta < 1:
        raise LoopError(f"accept_delta must be >= 1 (a tie must not churn the champion), got {accept_delta}")
    if confirm_every < 1:
        raise LoopError(f"confirm_every must be >= 1, got {confirm_every}")
    if epoch_len < 1:
        raise LoopError(f"epoch_len must be >= 1, got {epoch_len}")
    if n_drafts < 1:
        raise LoopError(f"n_drafts must be >= 1, got {n_drafts}")

    started = clock()
    concurrency = max_concurrency or cfg.max_concurrency
    total_cost = 0.0

    lockbox: Lockbox | None = None
    if lockbox_dir is not None:
        try:
            lockbox = read_lockbox(lockbox_dir)
            check_disjoint(tasks, lockbox)
        except FoldError as err:
            raise LoopError(str(err)) from err
    elif not allow_no_lockbox:
        raise LoopError(
            "no lockbox given — an evolution run makes hundreds of selections, all optimistic; the "
            "lockbox is the only number that is not. Run `acumen lockbox` and pass --lockbox, or "
            "pass --no-lockbox to proceed without a generalisation claim"
        )

    selection: HeadroomSelection | None = None
    if headroom_only:
        diffs = screen(runs_root, tasks, by_model=True)
        selection = select_headroom(diffs, tasks, split="test", models=cfg.models)
        if on_select is not None:
            on_select(selection)
        if not selection.selected:
            raise LoopError("no task has headroom to evolve on — run `acumen bench --no-skill` on more tasks first")
        tasks = selection.selected

    by_id = {t.id: t for t in tasks}
    pool_ids = sorted(by_id)
    deny_dirs: list[Path] = [runs_root / LOCKBOX_RUNS_DIRNAME, runs_root / DRAFTS_DIRNAME]
    if lockbox is not None:
        deny_dirs.append(lockbox.directory)

    # Seed: rulebook v1 and its skill. The champion chain starts here.
    seed_version = rb.seed_default(rulebooks_root)
    if seed_version != "v1":
        raise LoopError(f"expected the rulebook chain to start at v1, found {seed_version}")
    champion = confirmed = "v1"
    accepts_since_confirm = 0
    journal_path = runs_root / JOURNAL_FILE
    journaled = _read_journal(journal_path)
    history: list[Generation] = []
    reason = f"reached the generation budget ({generations})"

    async def ensure(version: str) -> tuple[Skill, float]:
        return await _ensure_skill(
            cfg=cfg,
            target=target,
            skills_root=skills_root,
            rulebook_text=rb.load_rulebook(rulebooks_root, version).text,
            expect_version=version,
            auth_mode=auth_mode,
            rationale=f"drafted from rulebook {version}",
            log_dir=log_dir,
            stream=stream,
        )

    async def bench_on(version: str, task_ids: Sequence[str] | None, splits: Sequence[str]) -> Score:
        skill, cost = await ensure(version)
        planned = await _bench(
            cfg=cfg,
            target=target,
            skill=skill,
            tasks=tasks,
            runs_root=runs_root,
            splits=list(splits),
            auth_mode=auth_mode,
            task_ids=list(task_ids) if task_ids is not None else None,
            max_concurrency=concurrency,
            on_start=on_bench_start,
            on_done=on_bench_done,
        )
        nonlocal total_cost
        total_cost += cost
        return score(runs_root, [p for p in planned if p.key.split == "test"])

    async def confirm(challenger: str, incumbent: str) -> bool:
        """Full-bench the challenger and incumbent; True promotes the challenger."""
        challenger_full = await bench_on(challenger, None, ["test"])
        incumbent_full = await bench_on(incumbent, None, ["test"])
        return challenger_full.passed > incumbent_full.passed

    for g in range(1, generations + 1):
        if max_wallclock_s is not None and clock() - started >= max_wallclock_s:
            reason = f"wall-clock cap of {max_wallclock_s:.0f}s reached before generation {g}"
            break
        gen_cost_before = total_cost
        directive = directive_for(g)
        epoch = (g - 1) // epoch_len
        subset = screen_subset(pool_ids, screen_size, epoch, seed)
        subset_tasks = [by_id[i] for i in subset]
        candidate_version = f"v{g + 1}"
        parent = champion

        # 1. The champion on this epoch's screen: train (evidence) + test (the bar).
        champion_screen = await bench_on(parent, subset, SPLITS)
        champion_skill, _ = await ensure(parent)

        # 2. Improve FROM THE CHAMPION under this generation's directive; draft the candidate.
        composed = directive if not feedback else f"{directive}\n\nMaintainer feedback: {feedback}"
        rulebook = await _ensure_rulebook(
            cfg=cfg,
            target=target,
            rulebooks_root=rulebooks_root,
            runs_root=runs_root,
            benched_skill=champion_skill,
            tasks=subset_tasks,
            baseline_version=parent,
            baseline_text=rb.load_rulebook(rulebooks_root, parent).text,
            auth_mode=auth_mode,
            feedback=composed,
            log_dir=log_dir,
            stream=stream,
            deny_dirs=deny_dirs,
            log_name=f"evolve-rulebook-{candidate_version}",
            expect_version=candidate_version,
        )
        total_cost += rulebook.cost_usd

        # 3. Screen the candidate on the identical subset; accept only past the noise floor.
        candidate_screen = await bench_on(candidate_version, subset, ["test"])
        accepted = candidate_screen.passed - champion_screen.passed >= accept_delta
        if accepted:
            champion = candidate_version
            accepts_since_confirm += 1

        # 4. The ratchet: every confirm_every accepts, a full bench decides — or reverts.
        confirm_ran = accepted and accepts_since_confirm >= confirm_every
        confirm_promoted: bool | None = None
        if confirm_ran:
            confirm_promoted = await confirm(champion, confirmed)
            if confirm_promoted:
                confirmed = champion
            else:
                champion = confirmed
            accepts_since_confirm = 0

        record = Generation(
            index=g,
            directive=directive,
            parent=parent,
            candidate=candidate_version,
            subset=tuple(subset),
            champion_screen=champion_screen,
            candidate_screen=candidate_screen,
            accepted=accepted,
            confirm_ran=confirm_ran,
            confirm_promoted=confirm_promoted,
            confirmed=confirmed,
            cost_usd=total_cost - gen_cost_before,
        )
        history.append(record)
        if g not in journaled:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with journal_path.open("a") as fh:
                fh.write(json.dumps(record.to_json()) + "\n")
            journaled.add(g)
        if on_generation is not None:
            on_generation(record)

    # Pending screen accepts are not left dangling: the run's champion is always confirmed.
    if champion != confirmed:
        if await confirm(champion, confirmed):
            confirmed = champion

    lockbox_drafts = baseline_drafts = None
    if evaluate_lockbox and lockbox is not None:
        lock_tasks = load_lockbox_tasks(lockbox)
        common = {
            "cfg": cfg,
            "target": target,
            "skills_root": skills_root,
            "rulebooks_root": rulebooks_root,
            "runs_root": runs_root,
            "n_drafts": n_drafts,
            "tasks": lock_tasks,
            "auth_mode": auth_mode,
            "max_concurrency": concurrency,
            "log_dir": log_dir,
            "stream": stream,
            "on_bench_start": on_bench_start,
            "on_bench_done": on_bench_done,
        }
        baseline_drafts = await _lockbox_eval(version="v1", **common)
        lockbox_drafts = baseline_drafts if confirmed == "v1" else await _lockbox_eval(version=confirmed, **common)

    return EvolveRun(
        generations=history,
        champion=confirmed,
        stopped_because=reason,
        journal_path=journal_path,
        lockbox_drafts=lockbox_drafts,
        lockbox_baseline_drafts=baseline_drafts,
        selection=selection,
        cost_usd=total_cost,
    )
