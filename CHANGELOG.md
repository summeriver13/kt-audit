# Changelog

All notable changes to this project are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `metric_floor`: accuracy compared with always answering the majority label of the same rows.
- `coverage`: expected keys (`--expect`), expected row count (`--expect-rows`), the reported `n`
  (`reported_n`), and empty or `ERROR` predicted labels.
- Classification files: a `y_pred_label` column or a non-numeric label column switches the file to
  accuracy, macro-F1 and kappa. `AuditConfig.task` forces either mode.
- A percentile bootstrap interval for every metric in `metrics[split]["intervals"]`; the
  `## Metrics` table prints the interval next to the value and the floor.
- `compare --metric {auc,acc,macro_f1,kappa}`.
- `AuditConfig` fields `task`, `metric_floor`, `floor_warn_margin`, `coverage`, `expect_keys`,
  `expect_rows`, `reported_n`, `reported_acc`.
- `AuditConfig.acc_tolerance` (0.01) so the reported-accuracy check no longer borrows
  `auc_tolerance`.
- `audit_path(..., expect=...)`; `audit_frame(..., expect=..., expect_digest=...)`; `input.expect`
  records the size and hash of the expected set.
- Examples `multiclass.csv` and `dropped_rows.csv`; `docs/rfc-0.3.md`;
  `docs/case-study-longtutor.md`.

### Fixed

- `cheap_baseline` skipped a fold whose training split holds a single class or whose labels
  fall outside 0/1; both crashed with `IndexError` and left no report.
- A label outside 0/1 leaves AUC empty instead of raising from `roc_auc_score`.
- `compare` accepted a non-finite prediction on the candidate side and then crashed with
  `IndexError`; both sides are checked and the sample sizes must match.
- `degenerate` applies the rare-minority rule to any number of classes, not only two.
- `coverage` warns when the expected set lacks a key column instead of comparing a partial key.

### Changed
- `schema_version` is `0.3`; fields were added inside `metrics[split]`.
- `comparison.json` names the `metric` and prints `value_a` / `value_b` instead of `auc_a` /
  `auc_b`.
- `degenerate` reports "no numeric predictions" instead of a missing column when the file carries
  labels only.
- `--no-alignment` drops the alignment findings only; the `## Metrics` table stays.

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
