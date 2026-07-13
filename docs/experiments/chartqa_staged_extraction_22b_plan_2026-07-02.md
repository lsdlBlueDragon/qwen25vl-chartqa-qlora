# ChartQA Module 22B staged chart-to-table extraction plan - 2026-07-02

## Scope

Module 22B runs staged chart-to-table extraction on the same all-wrong diagnostic subset.

It does not run full validation and does not train a new LoRA adapter.

By default, it skips the 8 high-priority `exclude_or_fix_reference` samples from Module 22A:

```text
12, 14, 105, 158, 470, 918, 1351, 1561
```

So the main diagnostic denominator is:

```text
77 samples
```

## Why this module exists

Module 21 showed:

- one-shot `table_json_only`: 14/85 = 16.47%;
- Module 21 oracle: 26/85 = 30.59%;
- after excluding reference/evaluator high-risk samples: 24/77 = 31.17%.

The one-shot chart-to-JSON route had signal but was not reliable enough. Module 22B tests whether staged extraction improves control and diagnosis.

## Stages

The new script is:

```text
scripts/run_chartqa_staged_extraction_diagnostic.py
```

It writes append-only JSONL outputs:

```text
outputs/chartqa_all_wrong_diagnostics/staged_extraction/
  overview.jsonl
  axes_legend.jsonl
  data_table.jsonl
  staged_table_json_only.jsonl
  staged_image_plus_table_json.jsonl
  staged_table_json_only_evaluated.jsonl
  staged_image_plus_table_json_evaluated.jsonl
  staged_extraction_summary.json
  staged_extraction_report.md
```

Stages:

1. `overview`: chart type, title, units, layout, likely relevant regions.
2. `axes_legend`: x/y axes, ticks, date/category order, scale notes, legend-color mapping.
3. `data_table`: question-relevant table and candidate values.
4. `staged_table_json_only`: answer using staged extraction only.
5. `staged_image_plus_table_json`: answer using image plus staged extraction.

## Recovery behavior

All long stages are rerunnable:

- Each stage appends one JSONL row per `sample_index`.
- Existing rows are skipped.
- If local output is missing but Drive output exists, the script restores from Drive first.
- Local and Drive JSONL outputs are written incrementally.
- Progress bars are shown for every extraction and QA stage.

## Success criteria

Compare Module 22B against Module 21:

| metric | previous reference |
|---|---:|
| one-shot `table_json_only` | 14/85 |
| one-shot valid JSON | 66/85 |
| Module 21 valid oracle after 22A exclusion | 24/77 |

Questions:

1. Does staged extraction improve JSON validity over one-shot extraction?
2. Does `staged_table_json_only` beat one-shot `table_json_only` after using the 77-sample valid denominator?
3. Does `staged_image_plus_table_json` show image-verification benefit?
4. Which reviewed_primary categories still fail after staged extraction?
5. Should the next step be prompt/schema refinement, crop-based extraction, or evaluator cleanup?

## Expected interpretation

If staged extraction improves recovered samples, continue with staged extraction and possibly add crop-specific stages.

If staged extraction does not improve recovered samples but JSON validity improves, inspect whether the data table is still missing exact values.

If staged extraction is worse than one-shot extraction, keep Module 21 as the current best diagnostic and redesign prompts before running more samples.

