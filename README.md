# kt-audit

[![PyPI](https://img.shields.io/pypi/v/kt-audit)](https://pypi.org/project/kt-audit/)
[![CI](https://github.com/summeriver13/kt-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/summeriver13/kt-audit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/kt-audit)](https://pypi.org/project/kt-audit/)

**Before you trust an AUC, audit the evaluation.**

`kt-audit` is a small, opinionated audit tool for knowledge-tracing (KT) experiments. It does not train models. It checks whether the evaluation setup behind a reported AUC is trustworthy.

Status: **alpha (0.3.x)**. The scope is intentionally narrow; see [docs/charter.md](https://github.com/summeriver13/kt-audit/blob/main/docs/charter.md) for the scope and non-goals.

## Why

KT papers and libraries report AUC, but the evaluation protocol is often under-specified:

- Are train events strictly before validation/test events for each user?
- Does the same item appear in train and test?
- When an item has multiple knowledge components (KCs), are some KCs train-only and others test-only?
- Do the reported metrics match a recomputation from the released predictions?

`kt-audit` turns those questions into a machine-readable report you can attach to a paper artifact, a PR, or a model release.

Two 2026 measurements motivated the v0.2 checks. In a tutoring benchmark, a depth-4 tree over seven
scalar features reproduced the gold diagnosis labels at 82.3 accuracy while the best LLM reached
42.0: the metric was measuring rule-following. In the same benchmark, a 3,000-token output budget
silently truncated 14.5% of samples and cost 7 points of score.

These are not hypothetical. In pyKT, a widely used KT benchmark library, the authors themselves warn that "the training and test data has overlap and the experimental results cannot be compared" ([pykt-team/pykt-toolkit#91](https://github.com/pykt-team/pykt-toolkit/issues/91)), and `data_process.py` points `valid` and `test` at the same file.

## Install

```bash
pip install kt-audit
```

Or with uv:

```bash
uv add kt-audit
```

From source:

```bash
git clone https://github.com/summeriver13/kt-audit
cd kt-audit
uv sync
uv run kt-audit --help
```

Requires Python 3.10+. Pin a specific version for reproducible paper artifacts.

## Quickstart

Download the synthetic example and run the audit:

```bash
curl -fsSLO https://raw.githubusercontent.com/summeriver13/kt-audit/main/examples/leaky.csv
kt-audit run leaky.csv --out report
```

Or, from a source checkout:

```bash
uv run kt-audit run examples/leaky.csv --out report
```

Output:

```text
verdict: fail
findings: 4 error, 0 warn
report: report/report.md
```

The command writes:

- `report.json` — stable, machine-readable.
- `report.md` — human-readable, with examples and limitations.

Exit codes: `0` pass/warn, `1` fail, `2` input/config error.

## What it checks

| Check | Catches | Default |
| --- | --- | --- |
| `temporal_split` | a user's train events after that user's evaluation events | error |
| `multi_kc` | same item with train-only and test-only KCs | error |
| `alignment` | missing/duplicate/out-of-range predictions; constant predictions; reported AUC or accuracy mismatch | error / warn |
| `cheap_baseline` | reported predictions that a tree over past-event aggregates matches or beats | error / warn |
| `degenerate` | predictions too few, too narrow, on too small a sample, or with a rare minority class | warn / info |
| `metric_floor` | an accuracy that does not beat always answering the majority label | error / warn |
| `coverage` | rows or keys the evaluation should have scored and did not | error / warn |

The report also gives every metric an interval. A difference smaller than that interval is noise,
whatever the table says.

Each finding has a `severity`, a `count`, and up to three examples.

### `cheap_baseline`

KT claims that long history adds information. This check tests that claim directly: it builds
eight features from **past events only** — prior count, prior accuracy, gap since the last event,
same-item history, item popularity, KC history, and sequence position — trains a depth-4 decision
tree on the training split, and compares its AUC with the reported predictions on the same rows.

- baseline AUC ≥ model AUC: `error`. The model adds nothing over cheap aggregates.
- model AUC − baseline AUC < `baseline_warn_margin` (0.02): `warn`.
- fewer than 50 labelled training rows, 30 evaluation rows, or one class: skipped.
- The comparison runs on the test split, or on valid when test is absent.

The numbers land in `report.json` under `baselines` and in a `## Baseline` table.

### `degenerate`

| Rule | Threshold | Why |
| --- | --- | --- |
| few distinct values | fewer than 10, at least 30 rows | predictions collapse to a few levels |
| narrow band | IQR below 0.02, at least 30 rows | no discrimination; AUC sits near 0.5 |
| small sample | fewer than 30 rows | interval too wide to compare (info, not a defect) |
| rare minority | minority share below 1% | AUC unstable |

Thresholds are module constants, not config keys. Change them with a release.

### `metric_floor`

A table of accuracies can sit entirely below a constant answer. The check compares accuracy with
the majority class of the same evaluation rows, and names that class:

```text
ERROR · metric_floor
test accuracy 0.283 does not beat always answering Procedural Error (0.506)
```

- accuracy below the majority share: `error`.
- accuracy within `floor_warn_margin` (0.02) above it: `warn`.
- `reported_acc` below the floor: `error`.
- The floor is computed from the evaluated rows, not from train: it is what the test labels allow.

See [docs/case-study-longtutor.md](https://github.com/summeriver13/kt-audit/blob/main/docs/case-study-longtutor.md)
for a run where every model of a paper lands there.

### `coverage`

A score is only comparable when you know what it scored. `coverage` compares the input with an
expected set and with the n the report claims:

```bash
kt-audit run predictions.csv --expect dataset.csv --expect-rows 1000
```

| Condition | Severity |
| --- | --- |
| an expected key is absent from the input | error |
| `reported_n` differs from the row count | error |
| a predicted label is empty or starts with `ERROR` | error |
| the input holds keys the expected set does not | warn |
| `--expect-rows` differs from the row count | warn |

Without `--expect` and without a row count the check is silent. An empty label is what a truncated
LLM call writes; that is the 14.5% loss the Why section describes.

### Labels and tasks

A file with `y_true` in {0, 1} and a numeric `y_pred` is a **ranking** file: it reports AUC,
accuracy, macro-F1 and kappa. A file with a `y_pred_label` column, or with string labels, is a
**classification** file: it reports accuracy, macro-F1 and kappa, and skips AUC and
`cheap_baseline`. Set `task` in the config to force either mode.

## Compare two runs

Two models that differ by 3 points on 200 rows may be indistinguishable. `compare` pairs the two
prediction files on their interaction keys and bootstraps the delta of the chosen metric on the
same resampled rows:

```bash
kt-audit compare baseline.csv candidate.csv --out comparison
kt-audit compare baseline.csv candidate.csv --metric acc --out comparison
```

`--metric` takes `auc` (the default), `acc`, `macro_f1` or `kappa`. The label metrics need a
predicted label on both sides; `auc` needs binary labels and refuses a classification file.

```text
verdict: b_better
delta: +0.4736 (95% CI [+0.3637, +0.5907])
report: comparison/comparison.md
```

- CI contains 0: `tie`. The sample cannot tell the two runs apart.
- Output: `comparison.json` and `comparison.md`.
- Defaults: 2000 rounds, seed 0, so the interval is reproducible.
- Inputs must share keys and agree on `y_true`; otherwise the command exits with code 2.

## Case study

`docs/case-study-assist09.md` runs the tool on ASSISTments 2009 with a per-user chronological split.
The v0.1 checks stay silent, which is correct; the cheap baseline reaches 0.7138, so a reported
0.771 passes and a reported 0.70 fails. Three severities changed as a result.

`docs/case-study-longtutor.md` runs the tool on a published diagnosis benchmark whose labels come
from a rule pipeline. Every model of that paper scores below the majority class; the floor check
says so in one line, and the intervals show that the differences between models are noise.

## Provenance

`report.json` records `tool_version` and `input.sha256`, so an artifact stays readable after the
input file is regenerated. The Markdown report prints both in its header.

## Input

One row per interaction or prediction. CSV or JSONL.

| Column | Required | Notes |
| --- | :--: | --- |
| `user_id` | yes | non-null |
| `timestamp` | yes | ISO8601 or numeric; comparable within a user |
| `split` | yes | `train` / `valid` / `test` |
| `kc_ids` | yes | pipe-separated KCs or JSON array |
| `item_id` | no | needed for multi-KC check |
| `y_true` | no | 0/1 for a ranking file; any label for a classification file |
| `y_pred` | no | a score in [0,1], or a label; omit to skip metric checks |
| `y_pred_label` | no | the predicted label; present means the file is a classification file |
| `fold` | no | checks run per fold |
| `sequence_id`, `position` | no | reserved for sequence checks |

See [docs/data-contract.md](https://github.com/summeriver13/kt-audit/blob/main/docs/data-contract.md) for the full contract.

## Python API

```python
from kt_audit import AuditConfig, audit_path

report = audit_path("events.csv", AuditConfig(reported_auc=0.80))
print(report.verdict)
for finding in report.findings:
    print(finding.severity, finding.check, finding.count)
```

## Design constraints

- Runtime dependencies: `numpy`, `pandas`, `scikit-learn`.
- CLI uses `argparse`; config uses JSON.
- Each check is a pure function over a prepared `DataFrame`.
- Synthetic examples only; no dataset is bundled.
- Findings are heuristics, not causal proof.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest --cov=kt_audit --cov-report=term-missing
bash scripts/wheel-smoke.sh
```

## Release

See [docs/release.md](https://github.com/summeriver13/kt-audit/blob/main/docs/release.md) and [CHANGELOG.md](https://github.com/summeriver13/kt-audit/blob/main/CHANGELOG.md). TestPyPI and PyPI publish through GitHub Actions trusted publishing.

## Roadmap

- v0.2 (shipped): `cheap_baseline`, `degenerate`, `compare`, provenance fields. See `docs/rfc-0.2.md`.
- v0.3 (shipped): classification files, metric intervals, `metric_floor`, `coverage`, `compare --metric`. See `docs/rfc-0.3.md`.
- v0.4: `label_rule` (can a depth-4 tree over the file's own columns write the labels), offline `judge-audit` for saved LLM judgments, pyKT adapter, HTML report, duplicate interactions, sequence truncation.

## AI disclosure

AI tools are used in development. The maintainer reviews every check definition, test, and release. Findings generated by the tool are deterministic and do not call an LLM.

## Contributing

See [CONTRIBUTING.md](https://github.com/summeriver13/kt-audit/blob/main/CONTRIBUTING.md). Each minor version freezes its checks; new checks land in a later minor version.

## License

MIT. See [LICENSE](https://github.com/summeriver13/kt-audit/blob/main/LICENSE).
