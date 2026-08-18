"""The rulebook artifact — the versioned SKILL-generating instructions being optimized.

**This is the crude P6 prototype.** The eventual rulebook (per the design in ``diary.md``) mirrors
``skills.py`` fully: content-hashed, immutable, with a ``meta.json`` recording parent and rationale.
This version keeps only what the loop prototype needs to answer its feasibility question — *does
optimizing the rulebook move a held-out score?* — namely a versioned directory of one file.

A *rulebook* is the template text that the drafting agent's prompt is built from. Today
:data:`acumen.prompts.DRAFT_PROMPT` is a hardcoded constant; here it becomes ``rulebooks/vN/
rulebook.md`` — the same template, with the same ``{package}``/``{src}``/``{out}``/``{skill_name}``
/… placeholders — so the loop can version it, mutate it from evidence, and draft skills from each
version. The rulebook, not the skill, is the artifact the outer loop improves; skills become
intermediates drafted fresh from whichever rulebook version is on trial.

Deliberately omitted vs. the real P6 (add when the loop proves worth it): the content hash, the
``meta.json`` provenance chain, and any immutability enforcement beyond "never overwrite".
"""

from __future__ import annotations

from pathlib import Path

from acumen.prompts import DRAFT_PROMPT
from acumen.skills import version_name, version_number

#: The single file each rulebook version directory holds.
RULEBOOK_FILE = "rulebook.md"

#: Placeholders a rulebook template MUST keep, or the skill it drafts is unusable: ``{out}`` tells
#: the agent where to write the skill, ``{skill_name}`` the name its frontmatter must declare. The
#: outer-improve agent edits the template, so these are the ones worth guarding structurally rather
#: than trusting the prompt to preserve.
REQUIRED_PLACEHOLDERS = ("{out}", "{skill_name}")

#: The fields :func:`acumen.prompts.draft_prompt` fills a rulebook template with — used to check a
#: template ``.format()``s cleanly (no stray ``{placeholder}`` the drafter can't supply).
_TEMPLATE_FIELDS = ("package", "version", "src", "python", "out", "skill_name", "feedback")


class RulebookError(ValueError):
    """Raised when a rulebook directory or template is missing or malformed."""


def rulebook_dir(rulebooks_root: Path, version: str | int) -> Path:
    """Return the directory holding a rulebook version."""
    name = version_name(version) if isinstance(version, int) else version
    version_number(name)  # validate the shape
    return rulebooks_root / name


def available_versions(rulebooks_root: Path) -> list[str]:
    """Return the rulebook versions present, in numeric order."""
    if not rulebooks_root.is_dir():
        return []
    found = [
        p.name for p in rulebooks_root.iterdir() if p.is_dir() and (p / RULEBOOK_FILE).is_file() and _is_version(p.name)
    ]
    return sorted(found, key=version_number)


def _is_version(name: str) -> bool:
    try:
        version_number(name)
    except ValueError:
        return False
    return True


def latest_version(rulebooks_root: Path) -> str | None:
    """Return the highest rulebook version present, or ``None`` if there are none."""
    versions = available_versions(rulebooks_root)
    return versions[-1] if versions else None


def next_version(rulebooks_root: Path) -> str:
    """Return the version a new rulebook should be written as."""
    latest = latest_version(rulebooks_root)
    return version_name(version_number(latest) + 1 if latest else 1)


def validate_rulebook(text: str) -> None:
    """Fail loudly if a rulebook template could not produce a working draft prompt.

    Two checks, because the outer-improve agent rewrites this text and either mistake silently
    breaks the next draft: (1) it must ``.format()`` with exactly the fields the drafter supplies —
    a stray ``{something}`` raises ``KeyError`` at draft time deep in the SDK; (2) it must keep the
    load-bearing placeholders (:data:`REQUIRED_PLACEHOLDERS`), without which the drafted skill has
    nowhere to go or no name.

    Raises
    ------
    RulebookError
        If the template references an unknown field or dropped a required placeholder.
    """
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in text]
    if missing:
        raise RulebookError(
            f"rulebook is missing required placeholder(s) {missing} — a draft prompt built from it "
            "would not tell the agent where to write the skill or what to name it"
        )
    try:
        text.format(**dict.fromkeys(_TEMPLATE_FIELDS, "x"))
    except (KeyError, IndexError, ValueError) as err:
        raise RulebookError(
            f"rulebook has a placeholder the drafter cannot fill ({err}) — templates may use only "
            f"{{{', '.join(_TEMPLATE_FIELDS)}}}; escape any literal brace as {{{{ or }}}}"
        ) from err


def load_rulebook(rulebooks_root: Path, version: str | int) -> str:
    """Load and validate one rulebook version's template text."""
    directory = rulebook_dir(rulebooks_root, version)
    path = directory / RULEBOOK_FILE
    if not path.is_file():
        raise RulebookError(f"no such rulebook version: {path}")
    try:
        text = path.read_text()
    except OSError as err:
        raise RulebookError(f"cannot read {path}: {err}") from err
    validate_rulebook(text)
    return text


def write_rulebook(rulebooks_root: Path, version: str | int, text: str) -> Path:
    """Write a rulebook version's template, refusing to overwrite an existing one.

    Versions are immutable here as in ``skills.py`` — the loop always writes the next version.
    """
    validate_rulebook(text)
    directory = rulebook_dir(rulebooks_root, version)
    path = directory / RULEBOOK_FILE
    if path.exists():
        raise RulebookError(f"{path} already exists — rulebook versions are immutable and never overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def seed_default(rulebooks_root: Path) -> str:
    """Ensure ``rulebooks/v1`` exists, seeded from the built-in draft prompt; always return ``"v1"``.

    v1 is the current hardcoded :data:`acumen.prompts.DRAFT_PROMPT` verbatim, so the loop's first
    iteration reproduces exactly today's drafting behaviour — the baseline the rulebook is optimized
    away from. Idempotent: if v1 is already present it is left untouched. This returns the *baseline*
    (``"v1"``), not the latest version, so a resumed loop still anchors on the same baseline it
    started from rather than treating an already-improved version as the starting point.
    """
    if "v1" not in available_versions(rulebooks_root):
        write_rulebook(rulebooks_root, "v1", DRAFT_PROMPT)
    return "v1"
