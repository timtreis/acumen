# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [0.0.1dev]

### Added

- `acumen mine` and `tasks --candidates`: task generation from the package's own notebooks and
  tutorials, with executable ground-truth scripts.
- `acumen coverage`: benchmark and skill coverage of the target package's public API.
- `acumen warm`: dataset pre-fetch so benchmark passes don't download.
- `acumen screen`: cheap subset benchmark for accept/reject decisions.
- `acumen lockbox`: write-once, digest-verified hold-out set.
- `acumen loop`: unattended draft/bench/improve cycles with stopping rules, k-fold cross-validated
  version picking (`--cv`), N-draft scoring (`--drafts`) and a one-shot lockbox verdict.
- `acumen evolve`: generational improve-from-best with exploration directives, screens and a
  full-benchmark ratchet; `--islands K` adds independent evolutions plus cross-pollination.
- Content-hashed, immutable rulebooks (the instructions that generate a skill) as the optimized
  artifact, with provenance per version.
- Per-model difficulty strata and skill-load rate as first-class benchmark metrics.
- Session limits pause a run (exit 3) and resume from disk instead of being recorded as failures.
