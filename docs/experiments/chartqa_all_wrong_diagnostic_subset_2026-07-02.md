# ChartQA all-wrong diagnostic subset plan - 2026-07-02

This experiment defines Module 21 for the Qwen2.5-VL ChartQA QLoRA project.

## Scope

Run input-side diagnostics on the recommended all-runs-wrong subset only.

Do not run full validation in this module.

Primary input:

```text
chartqa_all_wrong_recommended_diagnostic_subset.csv
```

Expected subset size:

```text
85 samples
```

## Motivation

The full-val all-runs-wrong audit found 325 samples missed by all seven available 3B runs. The manual review refined the failure distribution:

| reviewed_primary | count |
|---|---:|
| `multi_step_calculation` | 95 |
| `numeric_value_or_scale` | 75 |
| `extreme_value_or_ranking` | 44 |
| `date_axis_reading` | 35 |
| `text_label_lookup` | 24 |
| `counting_or_category_count` | 19 |
| `visual_mapping_or_legend` | 17 |
| `yes_no_or_boolean` | 12 |
| `data_or_evaluator_issue` | 2 |
| `date_serial_or_label_format` | 2 |

Key review flags:

| review_flag | count |
|---|---:|
| `needs_table_or_value_extraction` | 298 |
| `shared_wrong_consensus` | 232 |
| `resolution_sensitive` | 211 |
| `needs_axis_date_grounding` | 130 |
| `needs_legend_color_mapping` | 93 |
| `calculation_after_extraction` | 66 |
| `data_or_evaluator_issue` | 16 |

The main project conclusion is that the remaining bottleneck is mostly chart structure/value extraction rather than another broad LoRA steps/rank variant.

## New source files

```text
scripts/prepare_chartqa_all_wrong_subset.py
scripts/run_chartqa_subset_ablation.py
scripts/run_chartqa_structured_extraction_diagnostic.py
scripts/summarize_chartqa_all_wrong_diagnostics.py
docs/colab_module21_all_wrong_diagnostics.md
```

## Drive layout

Module 21 uses this Drive root:

```text
/content/drive/MyDrive/qwen25vl-chartqa-qlora
```

Local Windows mirror:

```text
G:\我的云端硬盘\qwen25vl-chartqa-qlora
```

Recommended Drive directories:

```text
outputs/chartqa_all_wrong_diagnostics/
  inputs/
    chartqa_all_wrong_manual_audit_report.md
    chartqa_all_wrong_manual_audit_table.csv
    chartqa_all_wrong_manual_audit_table.json
    chartqa_all_wrong_recommended_diagnostic_subset.csv
  data/
    chartqa_all_wrong_diagnostic_subset_85.jsonl
    chartqa_all_wrong_diagnostic_subset_85_summary.json
  ablation_runs/
  structured_extraction/
  summaries/

scripts_module21/
  prepare_chartqa_all_wrong_subset.py
  run_chartqa_subset_ablation.py
  run_chartqa_structured_extraction_diagnostic.py
  summarize_chartqa_all_wrong_diagnostics.py
```

## Module 21 run order

After a fresh Colab runtime:

```text
1.1 -> 1.3 -> 1.4 -> 21.1 -> 21.2 -> 21.3 -> 21.4 -> 21.5 -> 21.6
```

## Diagnostic runs

High-resolution/prompt ablations:

| run | model path | max_pixels | prompt |
|---|---|---:|---|
| `baseline_maxpix_802816` | base 3B | 802816 | default |
| `hardmix_maxpix_602112` | hardmix adapter | 602112 | default |
| `hardmix_maxpix_802816` | hardmix adapter | 802816 | default |
| `f_maxpix_802816` | steps250/r16/a32 adapter | 802816 | default |
| `hardmix_axis_legend_prompt_802816` | hardmix adapter | 802816 | axis/legend grounding prompt |

Structured extraction diagnostics:

| run | purpose |
|---|---|
| `qwen25vl3b_chart_to_json_802816` | extract chart structure into JSON-like text |
| `table_json_only` | answer from extracted structure only |
| `image_plus_table_json` | answer from image plus extracted structure |

## Success criteria

The module should answer:

1. How many all-wrong subset samples are recovered by high resolution alone?
2. Does hardmix or steps250/r16/a32 work better on the hard subset?
3. Does explicit axis/legend grounding prompt recover additional samples?
4. Does chart-to-JSON/table-assisted QA recover more samples than image-only?
5. Which reviewed primary categories benefit most?

Core metrics:

```text
relaxed_correct
exact_match
recovered_indices
recovered_by_reviewed_primary
recovered_by_review_flags
still_wrong_after_all_diagnostics
```

## Rerun/recovery requirements

All scripts are designed for Colab disconnect/restart:

- restore inputs from Drive;
- restore helper scripts from Drive if not already present in the repo;
- restore required adapters from Drive;
- append prediction/extraction JSONL incrementally;
- skip completed `sample_index` rows on rerun;
- write outputs locally and copy them to Drive;
- use progress bars for dataset scan, adapter restore, prediction, extraction, and summary steps.

## Backlog: evaluator/data cleanup

This is intentionally not the primary Module 21 implementation.

Next module candidate:

```text
Module 22A - data/evaluator cleanup list
```

Backlog items:

- Excel serial date references, e.g. samples 12 and 14.
- Color granularity exact-match issues, e.g. `Light blue` vs `Blue`.
- list-vs-sum and list ordering ambiguity.
- question/reference answer-type mismatch.
- OCR near-match normalization for names and labels.
- `increase` vs `increasing` semantic equivalence.

These should be tracked separately so they do not inflate model failure counts.

