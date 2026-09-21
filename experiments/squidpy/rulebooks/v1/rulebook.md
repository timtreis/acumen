You are writing a Claude Skill for the Python package `{package}` (version {version}).

A skill is documentation written for an agent, not for a human. Its only purpose is to
make an agent that has never used `{package}` succeed at real tasks with it on the first
try. It is not a tutorial, not a README, and not a sales pitch.

The agent arrives with a goal stated in plain English — what a user wants done, not which
function to call. The skill's job is to get it from that goal to a working result: route it
to the right entry point and the right sequence of steps, so it never has to reverse-engineer
that from the module layout.

# What you can read

- The package's source is at `{src}`. Read it — the source, the docstrings, the examples,
  the docs directory. This is the ground truth about how the package behaves.
- `{package}` is also installed; run `{python}` to check anything you are unsure about.
  Verify claims before you write them down.

# What you must write

Your working directory is `{out}`. Write:

1. `{out}/SKILL.md` — required. It must begin with YAML frontmatter, exactly:

---
name: {skill_name}
description: <one sentence: what this skill covers and when to use it>
---

   The `name` must be exactly `{skill_name}`.

   The `description` is load-bearing and must be HONEST. It is the only part of the skill
   an agent sees before deciding whether to open it, and it is the only thing that gets
   the skill loaded at the right moment.
   State what the skill covers and the goals it applies to — name the outcomes a user would
   actually phrase, so the skill loads when one of them comes up. Do not oversell it, and do
   not claim coverage the body does not deliver. Use this format:
   [What it does] + [When to use it: the goals/triggers it fires on] + [Key capabilities]

2. `{out}/references/*.md` — optional. Use these for detail that only some tasks need.

# How to write it

- **Organize around goals, not modules.** Work out what people actually use `{package}` to
  accomplish — read its examples, tutorials, and docs, not just its API — and structure the
  skill so a stated goal maps to the right entry point and the right sequence of steps. Do not
  just mirror the package's module layout.
- **Write what is not guessable.** An agent already knows Python and can read a
  traceback. Spend your words on what it would get WRONG by guessing: non-obvious
  defaults, required preprocessing, the function that looks right but isn't, where
  results are written, argument shapes and orientation, footguns the API invites, right order
  of steps to follow.
- **Generalize within each goal.** Organize around categories of goal, but keep the guidance
  under each one general enough to cover any task in that category. Don't enumerate one-off
  recipes — that's the failure mode on the other side of module-mirroring.
- **Don't transplant the tutorial's incidental setup onto every task.** When you pin down a
  computation's recipe from the package's own worked example, separate two things: the
  sequence of calls and package-level conventions (which function, which mode/flavor argument,
  what order steps run in) versus that example's own incidental choices (a demo crop region or
  image subset, an illustrative gene-count cutoff, an extra denoise/smooth pass) that exist only
  because the tutorial needed one small, tidy example to show off. Write the first into the
  skill as the general recipe; do not write the second in as something to always do. A goal
  that doesn't mention restricting to a subset, cropping to a region, or an extra smoothing pass
  should run on the full object/image as given — adding a restriction or preprocessing step the
  task never asked for (recomputing and filtering to a gene subset, cropping to the region the
  docs example happened to use) silently changes the result. That is a wrong-answer trap, not a
  best practice, and it is easy to fall into precisely because it looks like careful, faithful
  reproduction of the package's own tutorial.
- **Test sensitivity, not just existence.** Some steps have several valid-looking entry points
  or optional-looking preprocessing — several interchangeable-seeming functions to build the
  same kind of input, an optional denoise/normalize/filter step before the main call, a choice
  of which subset of the data to run an expensive statistic over, a chunked/tiled execution
  parameter on an image-processing call (results can shift at tile boundaries versus processing
  the whole image at once), or a parameter that accepts either the package's own simplest
  built-in implementation or an external/specialized model plugged in as a callable. Don't just
  document that alternatives exist — actually run it more
  than one way on real data and check whether the result changes materially. If it does, name
  the choice that matches the package's own tutorials/examples for the case at hand, and say
  what goes wrong with the other choices — e.g. a generic built-in default tuned for a coarser
  version of the task (coarse thresholding instead of per-object detection) can run cleanly and
  return a plausible-looking but substantially wrong count or value. "Runs without an error" and
  "gives the right answer" are different claims; a step that silently degrades results without
  raising is a worse trap than one that crashes, because nothing prompts a second look.
- **Don't trust a precomputed column on a bundled example object.** Example datasets shipped by
  the package often already carry derived columns/flags (a "highly variable", "filtered", or
  "clustered" style column) baked in from whatever preprocessing built that object — possibly
  with different parameters than the ones a task actually states. If a task says to restrict to,
  filter by, or recompute such a category, recompute it fresh with the exact parameters the task
  gives rather than reusing the pre-existing column of the same name; note this explicitly if the
  object arrives with one, since silently reusing it is an easy, invisible mistake.
- **For any "rank X and report the top/strongest one" goal, pin down the exact recipe, don't
  approximate it.** Rankings (top gene by a spatial statistic, strongest pair by an enrichment
  score, object count from a segmentation) are exquisitely sensitive to inputs that look like
  free choices but aren't: which graph-construction function and neighbor parameters build the
  underlying graph, which subset of genes/cells/objects the ranking runs over (and in what
  order, if a subset is sliced rather than re-ranked), and which values are used for a
  normalize/threshold/smooth step feeding into it. A different but equally-reasonable-looking
  choice on any one of these commonly changes which item comes out on top, silently, with no
  error. Reproduce the package's own worked example for that goal end to end, on real data, using
  its exact functions and parameter values (not a function or parameter you independently judge
  to be equivalent or more modern), and write those exact steps into the skill. If you must
  deviate, state the deviation and confirm by running both ways that the top result is unchanged.
  This means the recipe's function sequence and its package-level parameters (mode, flavor,
  graph-construction settings) — not the example's own incidental literals such as a demo crop
  region or an illustrative subset size; carry those over only if the task itself states that
  number (see the transplant point above).
- **State the ordering convention when a result is a pair or a list of names.** If a goal's output
  names two or more categories (e.g. "the pair with the strongest enrichment"), say exactly how
  their order is determined — e.g. read directly off the underlying matrix/table's row and column
  labels in the order they appear there, without re-sorting — so the agent doesn't invent its own
  order (reversed pairs are a wrong answer even when the pair itself is right). If a task instead
  asks for alphabetical order, give the exact code idiom, not just the word "alphabetically":
  Python's plain `sorted()` on strings is case-sensitive (all capitals sort before all lowercase
  letters), so `sorted(["T cells", "endothelial"])` puts "T cells" first even though a person
  reading "alphabetical" would expect "endothelial" first. Tell the agent to sort with a
  case-insensitive key, e.g. `sorted(names, key=str.lower)`, and say this explicitly rather than
  trusting the agent to remember it — this exact mistake is easy to make and easy to miss.
- **Flag expensive defaults.** If the natural, unrestricted way to call a function — over every
  gene, every cell, with the default number of permutations/iterations — can take far longer
  than a task's time budget, say so and name the standard mitigation (a smaller gene/cell
  subset, fewer permutations, a faster related function). Don't leave an agent to discover a
  multi-minute hang by trial and error.
- **Progressive disclosure.** `SKILL.md` should be short and route to `references/` for
  depth. An agent pays for every token of it on every task, including the tasks where it
  is irrelevant. If `SKILL.md` is long, you are taxing every run.
- **Be concrete.** A correct short code example beats a paragraph of prose. Show the real
  call, with the arguments that matter.
- **When a goal's answer names several items, tell the agent to report them as plain,
  human-phrased text** (e.g. `A, B` or `A and B`), not a Python/JSON list literal (e.g.
  `['A', 'B']`) — unless the task explicitly asks for code syntax. An agent that builds the
  result as a list in code will often just print or write that list's `repr()`, which reads as
  a different string even when every item in it is correct.
- **Prefer removing text over adding it.** Anything that merely restates the obvious is
  worse than nothing: it costs tokens and buries the parts that matter.
- **No hedging.** Say what to do.

# Verify before you finish

Do not write claims you have not checked. If you assert a default value, a return type,
or where an output lands, confirm it in the source or by running `{python}`. A skill that
confidently states something false is worse than no skill at all — it will send an agent
in the wrong direction with full confidence.
{feedback}
When you are done, `{out}/SKILL.md` must exist and start with the frontmatter above.
