# Changelog

All notable changes to this project are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-26

### Added

- `cheap_baseline` check: a depth-4 tree over past-event aggregates. Errors when the reported predictions do not beat it, warns when they barely do.
- `degenerate` check: few distinct values, narrow band, small sample, rare minority class.
- `compare` subcommand: paired bootstrap of the AUC delta between two prediction files, 2000 rounds, seed 0.
- `tool_version` and `input.sha256` in `report.json`; a `## Baseline` table in `report.md`.
- `AuditConfig.cheap_baseline`, `AuditConfig.degenerate`, `AuditConfig.baseline_warn_margin`.
- Synthetic fixtures `examples/floor.csv`, `examples/degenerate.csv`, `examples/compare_a.csv`, `examples/compare_b.csv`, plus `scripts/make_examples.py`.
- `docs/rfc-0.2.md`.

### Changed

- `schema_version` is now `0.2`. Fields were added; nothing was renamed.
- `Report` gains `tool_version` and `baselines`; `audit_frame` gains a keyword-only `digest`.
- The version string moved to `_version.py`; `audit_frame` and `audit_path` share `read_table`.
- A small evaluation split produces an `info` finding, not a `warn`: sample size is a fact, not a defect.
- "Users appear in more than one split" is now `info`. A per-user chronological split puts every user in both splits and leaks nothing; a per-user chronology violation is still an `error`.
- "Global timestamps overlap across splits" is now `info` for the same reason.
- "The same item appears in train and test" is now `info`: normal in KT, a defect only for cold-start item evaluation.
- The three changes above come from `docs/case-study-assist09.md`; on the standard KT protocol the v0.1 checks are silent by design.

## [0.1.1] - 2026-10-21

### Fixed

- Corrected installation instructions now that the package is published on PyPI.
- Made documentation links absolute so they also work on the PyPI project page.
- Added PyPI, CI, and Python version badges.

### Changed

- No runtime or public API changes.

## [0.1.0] - 2026-10-20

### Added

- Initial public alpha.
- Temporal split, multi-KC leakage, and prediction alignment checks.
- CLI with JSON and Markdown reports.
- Python API: `audit_frame`, `audit_path`, `AuditConfig`, `Report`.
- TestPyPI and PyPI trusted publishing through GitHub Actions.
