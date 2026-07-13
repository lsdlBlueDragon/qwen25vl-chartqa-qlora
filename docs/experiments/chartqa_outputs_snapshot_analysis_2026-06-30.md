# ChartQA Outputs Snapshot Analysis

Date: 2026-06-30

Source folder:

```text
C:/Users/90553/Downloads/outputs-20260630T025707Z-3-001/outputs
```

## Contents

The snapshot contains:

- Six adapter folders:
  - `chartqa_qlora_smoke_100`
  - `chartqa_qlora_train1k_steps100`
  - `chartqa_qlora_train1k_steps200`
  - `chartqa_qlora_calcnum1k_steps100`
  - `chartqa_qlora_hardmix1k_steps100`
  - `chartqa_qlora_train1k_steps250_r16a32`
- Early sequential val100 results in `metrics`, `chartqa_adapter`, `chartqa_prompt_c`, and `analysis`.
- Fixed random/stratified val100 results in `chartqa_3b_new_benchmark`.
- 7B diagnostic runs in `chartqa_7b_baseline`, `chartqa_7b_diagnostics`, and related comparison files in `analysis`.
- Full ChartQA val1920 trained-adapter results in `chartqa_3b_full_val`.

The most reliable comparisons are:

1. Fixed random/stratified val100: `chartqa_3b_new_benchmark`
2. Full val1920 trained adapters: `chartqa_3b_full_val`

Older sequential/head100 results are useful for history but should not be mixed directly with the fixed val100 or full-val claims.

## Fixed Random/Stratified Val100

| run | exact | relaxed | numeric relaxed |
|---|---:|---:|---:|
| `baseline_default` | 65/100 = 65.00% | 74/100 = 74.00% | 50/82 = 60.98% |
| `standard_steps100` | 69/100 = 69.00% | 75/100 = 75.00% | 50/82 = 60.98% |
| `standard_numeric_final` | 68/100 = 68.00% | 75/100 = 75.00% | 50/82 = 60.98% |
| `experiment_a_steps200` | 70/100 = 70.00% | 76/100 = 76.00% | 51/82 = 62.20% |
| `experiment_b_calcnum` | 68/100 = 68.00% | 75/100 = 75.00% | 51/82 = 62.20% |
| `experiment_d_hardmix` | 70/100 = 70.00% | 75/100 = 75.00% | 50/82 = 60.98% |
| `experiment_f_steps250_r16a32` | 71/100 = 71.00% | 76/100 = 76.00% | 49/82 = 59.76% |

Val100 conclusions:

- Fine-tuned 3B adapters improve over the 3B baseline modestly: best relaxed is 76% vs baseline 74%.
- F is best exact on val100 and tied best relaxed.
- Steps200 and F are the strongest single-adapter val100 runs.
- Numeric-specific prompt/result variants did not produce a robust gain.

Val100 oracle and selector:

- Multi-run oracle before F: 80/100 = 80%.
- Multi-run oracle including F: 83/100 = 83%.
- Best simple selector on val100: `date_or_axis -> F; otherwise steps200`, reaching 80/100 relaxed.
- F vs steps200 on val100: improved 4, regressed 4, net relaxed gain 0.
- Among the 50 samples routed to F by the selector rule: F improved 4, regressed 0, both correct 38, both wrong 8.

## Full Val1920 Trained Adapters

| run | exact | relaxed | numeric relaxed | human relaxed | machine relaxed |
|---|---:|---:|---:|---:|---:|
| `standard_steps100` | 1317/1920 = 68.59% | 1483/1920 = 77.24% | 981/1478 = 66.37% | 689/960 = 71.77% | 794/960 = 82.71% |
| `experiment_a_steps200` | 1325/1920 = 69.01% | 1489/1920 = 77.55% | 978/1478 = 66.17% | 690/960 = 71.88% | 799/960 = 83.23% |
| `experiment_b_calcnum` | 1323/1920 = 68.91% | 1487/1920 = 77.45% | 977/1478 = 66.10% | 683/960 = 71.15% | 804/960 = 83.75% |
| `experiment_d_hardmix` | 1331/1920 = 69.32% | 1495/1920 = 77.86% | 980/1478 = 66.31% | 693/960 = 72.19% | 802/960 = 83.54% |
| `experiment_f_steps250_r16a32` | 1334/1920 = 69.48% | 1491/1920 = 77.66% | 964/1478 = 65.22% | 691/960 = 71.98% | 800/960 = 83.33% |

Full-val conclusions:

- Best exact single adapter: `experiment_f_steps250_r16a32` at 69.48%.
- Best relaxed single adapter: `experiment_d_hardmix` at 77.86%.
- The adapter spread is very small: 0.89 pp exact and 0.62 pp relaxed.
- Machine-generated questions are much easier than human questions: roughly 82.7%-83.8% relaxed vs 71.2%-72.2% relaxed.
- Full-val trained-adapter oracle: 1587/1920 = 82.66%.
- All-trained-adapters-wrong: 333/1920.

Important limitation: the full-val set currently has trained adapters only. There is no full-val 3B baseline and no full-val prompt-only control in this snapshot, so the full-val data does not yet establish a baseline-to-finetuned gain.

## Full-Val Error Pattern

| run | calculation | date/axis | numeric value | text/label | color/legend | yes/no | scale/unit |
|---|---:|---:|---:|---:|---:|---:|---:|
| `standard_steps100` | 142 | 114 | 87 | 67 | 14 | 13 | 0 |
| `experiment_a_steps200` | 141 | 112 | 89 | 63 | 13 | 13 | 0 |
| `experiment_b_calcnum` | 143 | 108 | 87 | 67 | 13 | 14 | 1 |
| `experiment_d_hardmix` | 144 | 106 | 86 | 63 | 12 | 12 | 2 |
| `experiment_f_steps250_r16a32` | 156 | 100 | 88 | 58 | 13 | 13 | 1 |

Interpretation:

- F improves exact accuracy and reduces date/axis and text/label errors.
- F has the worst calculation error count and the lowest numeric relaxed accuracy, which explains why it is not the best relaxed adapter.
- Hardmix is the best relaxed adapter because it has the best overall balance, not because it dominates every category.

## Full-Val Selector Check

Reapplying the val100 selector idea on full val:

| method | relaxed | exact |
|---|---:|---:|
| `experiment_a_steps200` only | 1489/1920 = 77.55% | 1325/1920 = 69.01% |
| `experiment_d_hardmix` only | 1495/1920 = 77.86% | 1331/1920 = 69.32% |
| `experiment_f_steps250_r16a32` only | 1491/1920 = 77.66% | 1334/1920 = 69.48% |
| `date_or_axis -> F; otherwise steps200` | 1497/1920 = 77.97% | 1333/1920 = 69.43% |
| `date_or_axis -> F; visual/calculation/extreme -> hardmix; otherwise steps200` | 1497/1920 = 77.97% | 1332/1920 = 69.38% |

The val100 selector pattern partially holds on full val, but the gain is small:

- `date_or_axis -> F; otherwise steps200` improves relaxed by +8 over steps200 and +2 over hardmix.
- The gain is diagnostic evidence of complementarity, not strong evidence for deploying a hand-written selector.

## Early Sequential Val100 and Prompt-C Results

Older sequential/head100 results show:

- Baseline sequential val100: 72% relaxed.
- Steps100 sequential val100: 73% relaxed.
- Later adapter variants on that old subset mostly cluster around 71% relaxed.
- Prompt-C variants did not improve: careful chart 70%, numeric final 71%.

These runs were useful for exploration but should be treated as superseded by the fixed random/stratified val100 and full-val results.

## 7B Diagnostic Results

Available 7B diagnostics:

| run | exact | relaxed |
|---|---:|---:|
| `chartqa_val_qwen25vl7b_baseline_100` | 48% | 70% |
| `chartqa_val_7b_random_default_512_100` | 39% | 75% |
| `chartqa_val_7b_random_default_1024_100` | 38% | 75% |
| `chartqa_val_7b_random_direct_512_100` | 39% | 72% |

7B comparison files show:

- 7B vs 3B baseline on the compared val100 subset: improved 8, regressed 8, net relaxed gain 0.
- 7B vs 3B standard steps100: improved 8, regressed 9, net relaxed gain -1.

Interpretation:

- The available 7B diagnostics do not justify switching to 7B as the next main path.
- Some 7B exact scores are much lower while relaxed is similar, which suggests answer formatting and output style differences are significant.
- A fair 7B decision would require a carefully controlled full-val or fixed-val run with the same prompt/evaluator, but current evidence does not make it the priority.

## Overall Conclusion

The current evidence supports a conservative story:

1. 3B QLoRA works, but the single-adapter gain is modest.
2. Small variants are tightly clustered; more steps/rank/data mixing did not produce a decisive winner.
3. F is best exact, hardmix is best relaxed.
4. Adapter complementarity is real: val100 oracle 83%, full-val trained-adapter oracle 82.66%.
5. The remaining bottleneck is likely chart grounding/readout and arithmetic, not simply LoRA capacity.
6. The full-val baseline is missing and should be run before making final baseline-vs-finetuned claims.

## Recommended Next Step

Highest priority:

1. Add/run full-val 3B baseline with the same module 19 evaluation pipeline.
2. Optionally add/run full-val `standard_numeric_final` as a prompt-only control.
3. Recompute full-val baseline-vs-hardmix and baseline-vs-F comparisons.
4. Recompute full-val oracle including baseline.

After that:

1. Do a stratified audit of the 333 all-trained-adapters-wrong full-val samples.
2. If the audit confirms grounding/readout failures, move to a small chart-to-table / derendering / OCR-assisted diagnostic instead of more blind QLoRA variants.
