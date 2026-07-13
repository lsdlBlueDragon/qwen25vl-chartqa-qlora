# ChartQA Module 23B Codex targeted hard-failure review - 2026-07-03

## Scope

This review completes the targeted diagnostics requested after Module 23A. It uses the clean-after-23A denominator and reviews only samples that remain wrong after answer normalization.

No GPU, model inference, training, or full-val run was used. Inputs are existing Module 21 / 22B / 23A outputs plus G-drive images.

## Starting Point

Module 23A clean denominator:

```text
67
```

Module 23A normalized oracle:

```text
30/67 = 44.78%
```

Remaining hard-failure queue before this targeted review:

```text
37
```

Hard-failure indices:

```text
18, 28, 29, 46, 189, 229, 241, 250, 251, 281, 290, 291, 310, 312, 326, 344, 381, 408, 424, 467, 523, 529, 571, 648, 667, 675, 688, 778, 781, 797, 804, 816, 877, 901, 976, 978, 1055
```

## Refined Diagnosis

The 37 remaining samples are not all true model hard failures.

| refined group | count | samples |
|---|---:|---|
| true hard failures | 28 | 28, 29, 189, 229, 250, 251, 281, 290, 291, 310, 312, 326, 344, 381, 408, 424, 467, 523, 529, 667, 675, 778, 781, 797, 804, 877, 901, 1055 |
| residual normalization candidates | 5 | 18, 241, 648, 688, 816 |
| residual reference candidates | 3 | 46, 571, 978 |
| residual question / answer-extraction candidate | 1 | 976 |

So the effective hard-failure set after 23B manual refinement is closer to:

```text
28/67
```

This is not a new official score yet; it is a diagnostic denominator recommendation.

## True Hard Failure Buckets

| bucket | samples | interpretation |
|---|---|---|
| legend / color / visual encoding binding | 28, 290, 312, 424 | Model fails to bind color or visual encoding to the right semantic label/value. |
| semantic filtering / counting / threshold count | 189, 251 | Model counts aggregate series or estimates instead of enumerating valid categories/years. |
| date-axis near-peak precision | 229, 250 | Model chooses a neighboring year when peaks are close. |
| spatial grounding | 326, 344 | Positional phrases like rightmost/middle/bottom are mapped to the wrong segment. |
| arithmetic / aggregation | 281, 291, 381, 467, 523 | Model needs to list operands first, then compute. |
| label after computation | 529, 667, 675, 778, 781, 877, 901 | Model must compute a target value/range/difference first, then return a label. |
| stacked bar total vs segment | 804 | Model reports a segment when the reference expects the stacked total. |
| multi-answer list | 29, 408, 797 | Model outputs one value/year instead of an exhaustive list. |
| crop/OCR visibility | 1055 | Label/value is truncated in the image; needs crop/zoom/OCR. |
| series vs total disambiguation | 310 | Model chooses total bar instead of a requested segment. |

## Residual Evaluator / Data Issues

These should be handled before using the 37 as a hard-failure benchmark:

| sample | issue |
|---:|---|
| 18 | prediction says blue / hex code while reference says Light blue; color-name normalization candidate |
| 46 | chart visibly shows 0.34 while reference is 0.0034; scale/reference mismatch |
| 241 | prediction sentence contains correct 900, but parser grabs the year 1990 first |
| 571 | chart shows Sudan (April 1983) as 150000, not reference 250000 |
| 648 | predictions say Increase/increases; reference is increasing |
| 688 | outputs often contain the correct two labels but order/plural/singular normalization misses them |
| 816 | prediction says blue / hex code while reference says light blue |
| 976 | question appears truncated; some output contains Danish citizens but with uncertainty |
| 978 | question asks software and ICT solutions in 2019, chart value is 7.7; reference 12.3 appears to be ICT services |

## Recommended Next Module

Do not train or full-val yet. The next useful module should be a targeted prompt/evaluator ablation over the 28 true hard failures:

1. `legend_table_prompt`: explicitly extract legend/color mapping before answering.
2. `operand_table_prompt`: force list of operands and computation expression before final answer.
3. `spatial_locator_prompt`: force row/column/segment localization for positional language.
4. `range_count_prompt`: force enumerate all categories/years satisfying a threshold/range condition.
5. `multi_answer_prompt`: force exhaustive list output when the reference is a list.

Separately, update evaluator normalization for:

- color synonyms and hex-to-color names;
- morphological trend words such as increase/increases/increasing;
- numeric answer extraction from sentences when the answer is near the end and the question contains years;
- list order plus singular/plural variants.

## Output Files

- `outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_hard_failure_queue.csv`
- `outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv`
- `outputs/chartqa_23b_hard_failure_diagnostics/contact_sheets/*.png`
- `docs/experiments/chartqa_23b_hard_failure_diagnostics_2026-07-03.md`
- `docs/experiments/chartqa_23b_codex_targeted_review_2026-07-03.md`
