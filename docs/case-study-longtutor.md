# Case study: a diagnosis benchmark whose labels are rules

What 0.3 says about a published diagnosis benchmark, and what it still cannot say.

## Setup

- Labels: 1,000 gold diagnosis rows of the LongTutor reproduction (ACL 2026). Four classes:
  Procedural Error 506, Recall Failure 212, Conceptual Gap 180, Transfer Deficit 102.
- Predictions: two saved annotation files, one per model, keyed by `_key`.
- Input: one row per gold row, `y_true` = gold diagnosis, `y_pred_label` = the model diagnosis.
- `user_id` = the user part of `_key`; `item_id` = the first 12 hex digits of the question hash.
- `timestamp` = row order, `split` = `test`, `kc_ids` = a constant. The label set carries no
  clock, no KC and no second split, so `temporal_split` and `multi_kc` have nothing to check here.
- The converted files are not shipped: the dataset license does not allow redistribution.

```bash
uv run python scripts/make_case_study_longtutor.py \
  --gold gold_split.jsonl \
  --predictions gpt-6-astra.jsonl \
  --out longtutor_astra.csv

kt-audit run longtutor_astra.csv --out report_astra
kt-audit run longtutor_deepseek_flash.csv --out report_flash
kt-audit compare longtutor_astra.csv longtutor_deepseek_flash.csv --metric acc --out compare
```

## Run 1: gpt-6-astra

```text
verdict: fail
findings: 1 error, 0 warn
```

```text
ERROR · metric_floor
test accuracy 0.283 does not beat always answering Procedural Error (0.506)
```

| Metric | Value | 95% CI | Floor |
| --- | ---: | --- | ---: |
| accuracy | 0.283 | [0.255, 0.311] | 0.506 |
| macro-F1 | 0.285 | [0.261, 0.307] | n/a |
| kappa | 0.146 | [0.119, 0.174] | n/a |

The rest of the report is `info`: `cheap_baseline` and `degenerate` skip a classification file.

## Run 2: deepseek-flash

| Metric | Value | 95% CI |
| --- | ---: | --- |
| accuracy | 0.325 | [0.298, 0.354] |
| macro-F1 | 0.325 | [0.299, 0.352] |
| kappa | 0.174 | [0.145, 0.205] |

```text
verdict: b_better
delta: +0.0420 (95% CI [+0.0220, +0.0620])
```

## Reading

1. **The whole table is below a constant answer.** The two models score 0.283 and 0.325 against a
   0.506 floor. The paper reports 13 models between 0.26 and 0.42: every one of them loses to
   always answering Procedural Error. One line of the report states it; the paper's table cannot.
2. **The gap between models is real but small.** +4.2 points of accuracy, CI [+2.2, +6.2]. Two runs
   of the same model family usually land inside such an interval, which is why `compare` exists.
3. **The report does not say why.** The source audit shows the gap comes from how often a model says
   Procedural Error: 62 times of 1,000 for deepseek-flash, 8 for gpt-6-astra, against 506 gold
   rows. More PE guesses, more accidental hits. `kt-audit` reports the delta, not the cause.
4. **Kappa is the honest number here.** 0.146 and 0.174 against a floor of 0 for chance agreement:
   the models barely exceed agreement by luck on four classes.

## What the tool cannot see

The labels themselves. The reproduction shows that a depth-4 tree over seven scalar features
reproduces these gold labels at accuracy 82.3, while the best LLM of the paper reaches 42.0. The
label pipeline embeds the thresholds (> 80%, < 50%, > 4 days) in the prompt that generates the
label, so the label is a function of columns every model also sees.

A label file plus a prediction file cannot show that. A `label_rule` check — does a depth-4 tree
over the file's own columns reproduce the labels — is the 0.4 candidate that would, and the
control experiment of the reproduction gives its threshold: under a protocol that hides the rules,
the same features reach only 0.51–0.58.
