# ChartQA QLoRA Train1k Steps100

## Purpose

This run is the first small-scale experiment after the QLoRA smoke test. It checks whether a small adapter trained on 1,000 ChartQA SFT samples can improve over the base Qwen2.5-VL-3B baseline on the same val-100 evaluation slice.

## Setup

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Training method: QLoRA / LoRA on 4-bit base model
- Train data: first 1,000 `train` samples from `HuggingFaceM4/ChartQA`
- Eval data: first 100 `val` samples from `HuggingFaceM4/ChartQA`
- Training steps: 100
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: `2e-4`
- Runtime: Colab T4

## Artifacts

- Train JSONL: `data/processed/chartqa_train_sft_1000.jsonl`
- Val JSONL: `data/processed/chartqa_val_sft_100.jsonl`
- Adapter: `outputs/adapters/chartqa_qlora_train1k_steps100`
- Drive adapter backup: `/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/adapters/chartqa_qlora_train1k_steps100`
- Adapter predictions: `outputs/chartqa_val_qlora_train1k_steps100_100.jsonl`
- Adapter metrics: `outputs/chartqa_val_qlora_train1k_steps100_100_metrics.json`

## Results

| Run | Exact | Relaxed | Relaxed numeric |
| --- | ---: | ---: | ---: |
| Baseline val100 | 49.00% | 72.00% | 67.61% |
| QLoRA train1k steps100 val100 | 50.00% | 73.00% | 69.01% |

## Initial Takeaway

The latest full notebook rerun improved by 1 exact-match point and 1 relaxed-accuracy point on the val-100 slice. A prior run reached 52.00% exact and 74.00% relaxed, so this small experiment shows a positive but modest gain with normal run-to-run variance. The next decision should be based on error and delta analysis rather than immediately scaling training blindly.

## Evaluation Note

The first delta analysis exposed a weakness in the relaxed numeric metric: year/date answers such as `2006` vs `2018` could be counted as relaxed-correct because they are within the default 5% numeric tolerance. The evaluator now treats year/date questions, and plain four-digit year references, as strict for relaxed numeric matching. Re-run metrics after this change before reporting final numbers.

## Error Export Cell

Run this in Colab to generate detailed error records for analysis:

```python
%cd /content/qwen25vl-chartqa-qlora

RUN_NAME = "chartqa_val_qlora_train1k_steps100_100"

!python scripts/evaluate_predictions.py \
  --predictions outputs/{RUN_NAME}.jsonl \
  --metrics-output outputs/{RUN_NAME}_metrics.json \
  --errors-output outputs/{RUN_NAME}_errors.jsonl \
  --evaluated-output outputs/{RUN_NAME}_evaluated.jsonl
```

Then summarize error types:

```python
!python scripts/analyze_chartqa_errors.py \
  --errors outputs/chartqa_val_qlora_train1k_steps100_100_errors.jsonl \
  --output outputs/chartqa_val_qlora_train1k_steps100_100_error_analysis.json
```

To compare baseline and adapter behavior, first export a baseline evaluated file:

```python
!python scripts/evaluate_predictions.py \
  --predictions outputs/chartqa_val_baseline_100.jsonl \
  --metrics-output outputs/chartqa_val_baseline_100_metrics.json \
  --evaluated-output outputs/chartqa_val_baseline_100_evaluated.jsonl
```

Then run the comparison:

```python
!python scripts/compare_chartqa_runs.py \
  --baseline-evaluated outputs/chartqa_val_baseline_100_evaluated.jsonl \
  --adapter-evaluated outputs/chartqa_val_qlora_train1k_steps100_100_evaluated.jsonl \
  --output outputs/chartqa_val_qlora_train1k_steps100_100_vs_baseline.json
```

Keep detailed `errors` and `evaluated` files local during routine iteration. Copy them to Drive only when they are needed for long-term reporting.
