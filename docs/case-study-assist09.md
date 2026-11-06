# Case study: ASSISTments 2009

What the audit says about a standard KT setup, and what it says about the tool.

## Setup

- Data: ASSISTments 2009 Skill Builder (corrected file), 346,860 interaction rows.
- Filters: first attempts only (`original == 1`), non-empty `skill_id`; the first 2,000 users by id.
- Result: 162,042 rows, 1,791 users, 14,759 items, 142 KCs, 128,915 train / 33,127 test.
- The positive rate is 68.5%, so accuracy would sit near 0.69 for a constant predictor; AUC is the
  metric that can move.
- Split: per user, chronological 80/20. `order_id` stands in for time.
- Columns: `user_id` = user, `item_id` = problem, `kc_ids` = skill, `y_true` = correct, `position` = position.

```bash
uv run python scripts/make_case_study.py skill_builder_data_corrected.csv --out assist09.csv
uv run kt-audit run assist09.csv --out report
```

The converted file is not shipped: the dataset license does not allow redistribution.

## Run 1: no reported AUC

```text
verdict: pass
findings: 0 error, 0 warn
```

All findings are `info`:

| Check | Finding |
| --- | --- |
| `alignment` | `y_pred` is missing; metric checks skipped |
| `degenerate` | no numeric predictions; distribution checks skipped |
| `multi_kc` | the same item appears in train and test (9,174 of 14,759 items) |
| `temporal_split` | global timestamps overlap across splits |
| `temporal_split` | users appear in more than one split (1,791 users) |

The first result matters: **on the dominant KT protocol, the v0.1 checks are silent.** A per-user
chronological split puts every user in both splits and overlaps the global timeline by design; it
does not leak, because no user's training event follows that user's own evaluation event.

## Run 2: a reported AUC of 0.771

```bash
echo '{"reported_auc": 0.771}' > config.json
uv run kt-audit run assist09.csv --config config.json --out report
```

```text
| Fold | Cheap baseline AUC | Model AUC | Delta | n train | n eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| __all__ | 0.7138 | 0.7710 | +0.0572 | 128915 | 33127 |
```

Verdict: `pass`. Eight aggregate features reach 0.7138; the reported number sits 5.7 points above
them, so the claim survives the check. This is the useful case: the check is not trigger-happy.

## Run 3: a reported AUC of 0.70

```text
verdict: fail
findings: 1 error, 0 warn

### ERROR · cheap_baseline

reported predictions do not beat a cheap history baseline (count: 33127)
```

A reported 0.70 on this split is below what past-event aggregates reach without any sequence model.
That is a claim worth questioning before it is cited.

## What the tool learned

Three checks changed severity because of this run:

| Check | Was | Is | Why |
| --- | --- | --- | --- |
| global timeline overlap | error | info | per-user splits overlap the global timeline by construction |
| users in more than one split | warn | info | same reason: the user is the evaluation unit |
| same item in train and test | warn | info | normal in KT; only a defect for cold-start item evaluation |

The KC-level leak (an item with train-only and test-only KCs) stays an `error`. That one is not
explainable by protocol convention.

## Limitations

- `order_id` is an ordinal proxy for time, not a wall clock. A real timestamp would allow checking
  gaps and duplicate timestamps.
- The file has no predictions, so `alignment` and `degenerate` were not exercised. Run 2 and 3 use a
  reported number, not a re-computation.
- The 2,000-user cap changes the baseline slightly; a full run needs no cap but takes longer.
- First-attempt filtering is a choice. Scaffolded attempts are a different population.
