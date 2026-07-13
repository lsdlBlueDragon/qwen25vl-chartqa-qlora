# ChartQA full-val all-runs-wrong audit - 2026-07-01

This note summarizes the first model-assisted audit of the 325 ChartQA full-val samples that every available 3B run missed under relaxed accuracy.

Source Drive root:

```text
G:\我的云端硬盘\qwen25vl-chartqa-qlora
```

Input evaluated files:

- `outputs/chartqa_3b_full_val/chartqa_val_full_3b_baseline_default_1920_evaluated.jsonl`
- `outputs/chartqa_3b_full_val/chartqa_val_full_3b_standard_steps100_1920_evaluated.jsonl`
- `outputs/chartqa_3b_full_val/chartqa_val_full_3b_standard_steps100_numeric_final_1920_evaluated.jsonl`
- `outputs/chartqa_3b_full_val/chartqa_val_full_3b_steps200_1920_evaluated.jsonl`
- `outputs/chartqa_3b_full_val/chartqa_val_full_3b_calcnum1k_steps100_1920_evaluated.jsonl`
- `outputs/chartqa_3b_full_val/chartqa_val_full_3b_hardmix1k_steps100_1920_evaluated.jsonl`
- `outputs/chartqa_3b_full_val/chartqa_val_full_3b_steps250_r16a32_1920_evaluated.jsonl`

Generated local audit artifacts:

```text
outputs/chartqa_all_runs_wrong_audit/chartqa_full_val_all_runs_wrong_model_audit.json
outputs/chartqa_all_runs_wrong_audit/chartqa_full_val_all_runs_wrong_model_audit.csv
outputs/chartqa_all_runs_wrong_audit/chartqa_full_val_all_runs_wrong_model_audit_summary.json
outputs/chartqa_all_runs_wrong_audit/contact_sheets/
scripts/audit_chartqa_all_runs_wrong.py
```

## Method

The audit script reconstructs the all-runs-wrong set by taking samples where all seven evaluated runs have `eval_relaxed_correct == false`.

Each sample is labeled with:

- `primary_error_type`
- secondary `audit_tags`
- all seven model predictions
- question, reference answer, human/machine split, image path
- consensus note, e.g. whether most runs converged on the same wrong answer

The labels are a first-pass model-assisted taxonomy from the question, reference, predictions, and visual spot-checks through category contact sheets. They are suitable for project direction and triage, but individual samples should still be manually confirmed before being used as final paper/report ground truth.

## Reconstructed Count

The script reconstructed exactly:

```text
all-runs-wrong records: 325
```

This matches the final module 20 oracle count:

```text
all_runs_wrong_count: 325
```

## Primary Error Type Distribution

| primary_error_type | count | share |
|---|---:|---:|
| `multi_step_calculation` | 141 | 43.38% |
| `date_axis_reading` | 102 | 31.38% |
| `visual_mapping_or_legend` | 27 | 8.31% |
| `counting_or_category_count` | 27 | 8.31% |
| `numeric_value_or_scale` | 13 | 4.00% |
| `extreme_value_or_ranking` | 10 | 3.08% |
| `date_serial_or_label_format` | 2 | 0.62% |
| `text_label_lookup` | 2 | 0.62% |
| `yes_no_or_boolean` | 1 | 0.31% |

## Secondary Tag Signals

Many failures have overlapping causes. The most common secondary tags were:

| tag | count |
|---|---:|
| `numeric_value_or_scale` | 283 |
| `shared_wrong_consensus` | 232 |
| `multi_step_calculation` | 151 |
| `date_axis_reading` | 129 |
| `visual_mapping_or_legend` | 93 |
| `extreme_value_or_ranking` | 86 |
| `counting_or_category_count` | 79 |
| `unstable_across_runs` | 22 |

The most important signal is `shared_wrong_consensus`: 232/325 samples have at least 5 of 7 runs converging on the same wrong answer. This suggests the bottleneck is usually shared chart understanding or visual readout, not adapter-specific variance.

## Human/Machine Split

Overall all-runs-wrong split:

| split value | count |
|---|---:|
| `human_or_machine = 0` | 193 |
| `human_or_machine = 1` | 132 |

Primary category by split:

| primary_error_type | human_or_machine=0 | human_or_machine=1 |
|---|---:|---:|
| `multi_step_calculation` | 110 | 31 |
| `date_axis_reading` | 30 | 72 |
| `visual_mapping_or_legend` | 23 | 4 |
| `counting_or_category_count` | 20 | 7 |
| `numeric_value_or_scale` | 3 | 10 |
| `extreme_value_or_ranking` | 4 | 6 |
| `date_serial_or_label_format` | 2 | 0 |
| `text_label_lookup` | 0 | 2 |
| `yes_no_or_boolean` | 1 | 0 |

Interpretation:

- Human-authored failures are dominated by calculation/readout and visual mapping questions.
- Machine-generated failures are heavily concentrated in date/axis and exact value/ranking extraction.

## Representative Examples

`multi_step_calculation`:

- sample 13: "What's the total sum of peak points of all three lines?", reference `155`, all seven runs predicted `182`.
- sample 21: ratio of dissatisfied vs satisfied, reference `2.125`, six of seven runs predicted `0.19375`.
- sample 29: asks for two percentages, reference `[4, 9]`, all seven runs predicted the sum `13`.

`date_axis_reading`:

- sample 1: asks which year the blue/green difference is 1, reference `2018`, six of seven runs predicted `2005`.
- sample 229: asks which year recorded the highest cases, reference `2018`, all seven runs predicted `2016`.
- sample 233: asks whether Myanmar's 1997 value is the median, reference `yes`, all seven runs predicted `no`.

`visual_mapping_or_legend`:

- sample 18: "colour of oppose", reference `Light blue`, all seven runs predicted `Blue`.
- sample 28: dark grey segment, reference `Both`, six of seven runs predicted `Neither/Other`.
- sample 154: bar color resembling a fruit, reference `orange`, all seven runs predicted `none`.

`numeric_value_or_scale`:

- sample 46: death rate reference `0.0034`, all seven runs predicted `0.34`, a scale/decimal failure.
- sample 368: export value reference `4.5`, all seven runs predicted `4500`.

`date_serial_or_label_format`:

- samples 12 and 14 have references such as `44538` and `44207`, while predictions are human-readable dates. These look like possible Excel-date-style label/evaluator edge cases and deserve special handling before being treated as pure model failures.

## Main Conclusions

The 325 shared failures are not mainly prompt formatting failures.

The dominant pattern is:

1. read values from the chart;
2. align values to date/axis/legend/category;
3. perform a simple but brittle calculation;
4. output a compact answer.

The failure point is usually in steps 1-3. More broad QLoRA runs are unlikely to close most of this gap by themselves.

The contact-sheet review supports the same direction as the earlier val100 all-wrong audit: remaining errors are mostly chart grounding, scale/date/axis readout, visual mapping, and arithmetic composition.

## Recommended Next Work

1. Treat `date_serial_or_label_format` as a data/evaluator cleanup bucket and inspect those two rows manually.
2. Use the 325-row CSV as the seed for a smaller manually verified benchmark, e.g. 60-100 samples stratified by primary error type.
3. For module 21, run resolution ablation on a subset enriched for:
   - `date_axis_reading`
   - `numeric_value_or_scale`
   - `visual_mapping_or_legend`
4. For a later diagnostic, test chart-to-table/OCR/derendering assistance before scheduling another broad LoRA variant.

