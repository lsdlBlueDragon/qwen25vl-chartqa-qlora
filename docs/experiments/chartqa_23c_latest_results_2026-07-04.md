# ChartQA Module 23C latest results - 2026-07-04

## Notebook Status

The latest notebook `qwen25vl_3b_chartqa_qlora (1).ipynb` had 74 cells when the Module 23C results were inspected. After adding the dependency/reconnect supplement, it has 97 cells. Module 23C cells executed successfully:

- `23C.1` restore scripts, inputs, and adapter: executed.
- `23C.2` normalization v2 evaluator-only: executed.
- `23C.3` routed targeted prompt ablation: `returncode=0`.
- `23C.4` summary read: executed.

## Normalization v2

```text
before v2: 30/67 = 44.78%
after v2:  34/67 = 50.75%
gain: +4
recovered: 18, 241, 648, 816
```

Rules:

```text
color_synonym_or_hex: 18, 816
numeric_answer_in_sentence: 241
trend_morphology: 648
```

## Routed Targeted Prompt Ablation

The routed targeted prompt run completed on the 28 true-hard samples.

```text
total predictions: 28
unique samples: 28
oracle recovered: 1
recovered index: 344
```

By prompt:

```text
legend_table_prompt: 0/5
multi_answer_prompt: 0/3
operand_table_prompt: 0/15
range_count_prompt: 0/3
spatial_locator_prompt: 1/2
```

## Interpretation

The routed prompt ablation is mostly negative. Simple prompt routing does not solve the hard failures. The only recovered sample is `344`, a spatial-position grounding case, so the only clearly positive signal is for `spatial_locator_prompt`.

The remaining bottleneck is likely not final-answer wording. It is structured visual grounding: correct legend binding, operand extraction, spatial localization, range aggregation, and multi-answer enumeration.

## Operational Fix Added

Because the previous notebook module did not clearly document dependencies and reconnect behavior, a new supplement module was added:

```text
Module 23C-SUPPLEMENT - dependencies, reconnect, checkpoint, and latest results
```

It includes:

- fresh-runtime dependency install/check cell;
- reconnect/resume rules;
- checkpoint inspection cell;
- exact routed resume command;
- latest 23C result summary.
