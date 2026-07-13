# ChartQA QLoRA Experiment F: Train1k Steps250 R16 A32

Date: 2026-06-27

## Purpose

Experiment F tests whether a slightly longer train1k run with a larger LoRA rank improves over the previous best 3B adapter on the fixed random/stratified ChartQA val100 benchmark.

The main comparison target is `experiment_a_steps200`, because it was the strongest previous single 3B run on the fixed benchmark.

## Setup

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Training data: `data/processed/chartqa_train_sft_1000.jsonl`
- Evaluation data: `data/processed/chartqa_val_random_sft_100_seed42.jsonl`
- Benchmark: fixed random/stratified val100 seed42
- Max steps: 250
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: `2e-4`
- Seed: 42
- Min pixels: 50176
- Max pixels: 401408
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- 4-bit loading: enabled
- Gradient checkpointing: enabled

## Artifacts

- Adapter: `outputs/adapters/chartqa_qlora_train1k_steps250_r16a32`
- Drive adapter: `/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/adapters/chartqa_qlora_train1k_steps250_r16a32`
- Run name: `chartqa_val_random_3b_steps250_r16a32_100`
- Benchmark outputs:
  - `outputs/chartqa_3b_new_benchmark/chartqa_val_random_3b_steps250_r16a32_100.jsonl`
  - `outputs/chartqa_3b_new_benchmark/chartqa_val_random_3b_steps250_r16a32_100_metrics.json`
  - `outputs/chartqa_3b_new_benchmark/chartqa_val_random_3b_steps250_r16a32_100_errors.jsonl`
  - `outputs/chartqa_3b_new_benchmark/chartqa_val_random_3b_steps250_r16a32_100_evaluated.jsonl`
  - `outputs/chartqa_3b_new_benchmark/chartqa_val_random_3b_steps250_r16a32_100_error_analysis.json`

All benchmark outputs were copied to:

```text
/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_3b_new_benchmark
```

## Results

| Run | Exact | Relaxed | Relaxed correct | Notes |
| --- | ---: | ---: | ---: | --- |
| `baseline_default` | 65.00% | 74.00% | 74/100 | fixed new baseline |
| `standard_steps100` | 69.00% | 75.00% | 75/100 | previous standard adapter |
| `experiment_a_steps200` | 70.00% | 76.00% | 76/100 | previous best relaxed score |
| `experiment_f_steps250_r16a32` | 71.00% | 76.00% | 76/100 | best exact, tied best relaxed |

## Pairwise Comparisons

| Comparison | Net relaxed gain | Improved | Regressed | Both correct | Both wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
| F vs `baseline_default` | +2 | 6 | 4 | 70 | 20 |
| F vs `standard_steps100` | +1 | 3 | 2 | 73 | 22 |
| F vs `experiment_a_steps200` | 0 | 4 | 4 | 72 | 20 |

## Error Pattern

Experiment F top error counts:

| Error type | Count |
| --- | ---: |
| `calculation_error` | 7 |
| `date_or_axis_label_error` | 6 |
| `numeric_value_error` | 6 |
| `text_or_label_error` | 3 |
| `yes_no_error` | 2 |

Compared with `experiment_a_steps200`, F appears to reduce date/axis-label errors, but increases calculation and numeric-value errors. This explains why exact accuracy rises slightly while relaxed accuracy stays flat.

## Decision

Experiment F is complete.

It should be kept as a useful artifact because it gives the best exact score so far and ties the best relaxed score. However, it does not clearly beat `experiment_a_steps200` on the main relaxed metric. The result does not justify blindly increasing LoRA rank, steps, or model size.

Current label for reporting:

```text
best exact / tied best relaxed: experiment_f_steps250_r16a32
best relaxed with simpler setting: experiment_a_steps200
```

## Next Module Plan

The next useful module is not another F training variant. It should be a focused comparison module:

1. Compare F vs `experiment_a_steps200`.
2. Export the 4 improved and 4 regressed samples.
3. Classify whether changes are caused by date/axis reading, arithmetic, numeric extraction, legend/color mapping, or answer formatting.
4. Decide whether the next step should be:
   - repeat seed;
   - targeted data resampling;
   - a lightweight selector over existing adapters;
   - chart-to-table or derendering assistance.

This can become module 15 in the notebook.

## Follow-up Diagnostics

### Module 15: F vs Steps200

Module 15 compared `experiment_f_steps250_r16a32` with `experiment_a_steps200` at the sample level.

Main result:

| Bucket | Count |
| --- | ---: |
| F improved over steps200 | 4 |
| F regressed from steps200 | 4 |
| both correct | 72 |
| both wrong | 20 |
| net relaxed gain | 0 |

The improved samples were mostly date/axis/scale related. The regressed samples were mostly calculation, counting, or numeric extraction cases.

### Module 16: Lightweight Selector Diagnostic

Module 16 tested whether the observed adapter complementarity can be captured by simple question-type rules.

Best selector rule:

```text
if question type is date_or_axis:
    use experiment_f_steps250_r16a32
else:
    use experiment_a_steps200
```

Result on the fixed val100 benchmark:

| Method | Relaxed |
| --- | ---: |
| `experiment_a_steps200` only | 76/100 |
| `experiment_f_steps250_r16a32` only | 76/100 |
| lightweight selector | 80/100 |
| current multi-run oracle | 83/100 |

Selector stability check for the 50 samples routed to F:

| Bucket among selected F samples | Count |
| --- | ---: |
| F improved over steps200 | 4 |
| F regressed from steps200 | 0 |
| both correct | 38 |
| both wrong | 8 |

Interpretation:

- The single-adapter fine-tuning result remains modest but real: fixed val100 relaxed improves from `74%` baseline to `76%` for the best adapters, and F reaches the best exact score at `71%`.
- The selector's `80%` is a composition/diagnostic result, not a standalone fine-tuned model score.
- The selector result is still useful because it shows different adapters learned complementary behavior, especially F's advantage on date/axis/scale questions.

### Next Step

Module 17 should analyze the remaining all-runs-wrong samples. These are the cases not solved by baseline, individual adapters, or the current selector/oracle pool, and they are the best evidence for whether the next stage should be more LoRA training or chart-to-table/derendering/tool assistance.

### Module 17: All-runs-wrong Error Attribution

Module 17 exported and classified the samples that every current 3B run answered incorrectly under relaxed evaluation.

Summary:

| Item | Value |
| --- | ---: |
| Fixed benchmark size | 100 |
| Multi-run oracle relaxed | 83/100 |
| All-runs-wrong samples | 17 |
| Hard visual/reasoning failures by heuristic | 15/17 |

Heuristic failure categories:

| Category | Count |
| --- | ---: |
| `date_or_axis_reading` | 7 |
| `multi_step_calculation` | 2 |
| `shared_wrong_reading` | 2 |
| `numeric_value_or_scale` | 1 |
| `visual_mapping_or_legend` | 1 |
| `counting_or_category_count` | 2 |
| `visual_mapping_or_extreme_value` | 2 |

Important caveat: these are heuristic labels, not a manually verified taxonomy. Some samples need manual relabeling before being used in a report. For example, label/list extraction cases and comparison questions can be over-grouped by simple keyword rules.

Interpretation:

- Most remaining failures are not simple output-format mistakes.
- Many failures look like chart grounding problems: wrong year, wrong axis point, wrong colored series, wrong extreme value, or wrong scale/unit.
- Several predictions converge to the same wrong value across all adapters, which suggests shared visual/readout failure rather than adapter-specific overfitting.
- This weakens the case for continuing small LoRA variants such as more steps or larger rank.

Current recommendation:

1. Stop blind `steps/rank` scaling for 3B QLoRA.
2. Keep the single-adapter result as modest but real fine-tuning improvement: `74% -> 76% relaxed`, with F reaching `71% exact`.
3. Keep the selector result as complementarity evidence only: `80% relaxed`, not a standalone fine-tuned model score.
4. Next useful work is manual error-audit and report preparation, or a small chart-to-table / derendering / OCR-assisted diagnostic.

## Full Validation Follow-up: 2026-06-30

Module 19 evaluated all trained 3B QLoRA adapters on the full ChartQA validation split (`val1920`).

Experiment F remains the best exact adapter, but it is not the best relaxed adapter on full validation:

| run | exact | relaxed |
|---|---:|---:|
| `experiment_d_hardmix` | 1331/1920 = 69.32% | 1495/1920 = 77.86% |
| `experiment_f_steps250_r16a32` | 1334/1920 = 69.48% | 1491/1920 = 77.66% |
| `experiment_a_steps200` | 1325/1920 = 69.01% | 1489/1920 = 77.55% |

Updated interpretation:

- F is still the best exact single adapter.
- `experiment_d_hardmix` is the best relaxed single adapter.
- F reduces date/axis and text/label errors, but increases calculation errors and has the lowest numeric relaxed accuracy among the five trained adapters.
- Full-val results reinforce the conservative conclusion: small 3B QLoRA variants are tightly clustered, and further blind rank/step increases are unlikely to be the most productive next step.

See `docs/experiments/chartqa_full_val_3b_adapters_2026-06-30.md` for the full comparison.
