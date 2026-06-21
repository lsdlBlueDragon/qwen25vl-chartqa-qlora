# ChartQA Evaluation

## Purpose

`scripts/evaluate_predictions.py` evaluates JSONL files produced by baseline or future adapter runs.

It reports:

- strict exact match;
- relaxed numeric match with 5% tolerance;
- optional percent-scale equivalence such as `72` vs `0.72`;
- relaxed correctness;
- latency summary;
- grouped metrics by `human_or_machine`;
- optional evaluated records and error records when explicitly requested.

## Colab Cell

```python
%cd /content/qwen25vl-chartqa-qlora

!python scripts/evaluate_predictions.py \
  --predictions outputs/chartqa_val_baseline_20.jsonl \
  --metrics-output outputs/chartqa_val_baseline_20_metrics.json \
  --drive-metrics-dir /content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/metrics
```

This default test-stage command writes only a small metrics JSON and backs it up to Drive.

When detailed error inspection is needed, request those files explicitly:

```python
%cd /content/qwen25vl-chartqa-qlora

!python scripts/evaluate_predictions.py \
  --predictions outputs/chartqa_val_baseline_20.jsonl \
  --metrics-output outputs/chartqa_val_baseline_20_metrics.json \
  --errors-output outputs/chartqa_val_baseline_20_errors.jsonl \
  --evaluated-output outputs/chartqa_val_baseline_20_evaluated.jsonl
```

Do not copy detailed `errors` and `evaluated` files to Drive during routine tests. They can be regenerated from the prediction JSONL.

## Notes

Strict exact match is still useful, but it undercounts cases where the model is semantically correct with a minor formatting difference.

Examples from the current baseline:

- `Not too much/ not at all` vs `Not too much/not at all`: text normalization should count this as correct.
- `72` vs `0.72`: relaxed numeric can count this as a percent-scale match.
- `Blue` vs `Light blue`: this remains wrong because the color is less specific.
