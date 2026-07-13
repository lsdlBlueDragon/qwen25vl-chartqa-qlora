# ChartQA Full Validation: Trained 3B QLoRA Adapters

Date: 2026-06-30

## Scope

This record summarizes full ChartQA validation results for the trained `Qwen/Qwen2.5-VL-3B-Instruct` QLoRA adapters.

- Evaluation split: ChartQA `val`, full 1920 examples
- Data file in Colab: `data/processed/chartqa_val_full_sft_1920.jsonl`
- Group balance observed in module 19.1: 960 human questions and 960 machine-generated questions
- Local downloaded result folder:
  `C:/Users/90553/Downloads/chartqa_3b_full_val-20260630T021535Z-3-001/chartqa_3b_full_val`
- Drive result folder:
  `/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_3b_full_val`

This run covers trained adapters only. It does not include the 3B baseline or prompt-only variants such as `standard_numeric_final`.

## Full Val Results

| run | exact | relaxed | numeric relaxed | human relaxed | machine relaxed | relaxed errors |
|---|---:|---:|---:|---:|---:|---:|
| `standard_steps100` | 1317/1920 = 68.59% | 1483/1920 = 77.24% | 981/1478 = 66.37% | 689/960 = 71.77% | 794/960 = 82.71% | 437 |
| `experiment_a_steps200` | 1325/1920 = 69.01% | 1489/1920 = 77.55% | 978/1478 = 66.17% | 690/960 = 71.88% | 799/960 = 83.23% | 431 |
| `experiment_b_calcnum` | 1323/1920 = 68.91% | 1487/1920 = 77.45% | 977/1478 = 66.10% | 683/960 = 71.15% | 804/960 = 83.75% | 433 |
| `experiment_d_hardmix` | 1331/1920 = 69.32% | 1495/1920 = 77.86% | 980/1478 = 66.31% | 693/960 = 72.19% | 802/960 = 83.54% | 425 |
| `experiment_f_steps250_r16a32` | 1334/1920 = 69.48% | 1491/1920 = 77.66% | 964/1478 = 65.22% | 691/960 = 71.98% | 800/960 = 83.33% | 429 |

## Comparison With Fixed Val100

| run | val100 exact | full exact | exact delta | val100 relaxed | full relaxed | relaxed delta |
|---|---:|---:|---:|---:|---:|---:|
| `standard_steps100` | 69.00% | 68.59% | -0.41 pp | 75.00% | 77.24% | +2.24 pp |
| `experiment_a_steps200` | 70.00% | 69.01% | -0.99 pp | 76.00% | 77.55% | +1.55 pp |
| `experiment_b_calcnum` | 68.00% | 68.91% | +0.91 pp | 75.00% | 77.45% | +2.45 pp |
| `experiment_d_hardmix` | 70.00% | 69.32% | -0.68 pp | 75.00% | 77.86% | +2.86 pp |
| `experiment_f_steps250_r16a32` | 71.00% | 69.48% | -1.52 pp | 76.00% | 77.66% | +1.66 pp |

## Oracle Across Trained Adapters

- Oracle relaxed correct across the five trained adapters: 1587/1920 = 82.66%
- All-trained-adapters-wrong count: 333/1920
- Fixed val100 multi-run oracle was 83/100 = 83.00%, with 17 all-runs-wrong examples

The oracle result is stable across val100 and full val: there is real adapter complementarity, but the remaining shared failures are still large.

## Error Pattern

Relaxed error analysis by run:

| run | calculation | date/axis | numeric value | text/label | color/legend | yes/no | scale/unit |
|---|---:|---:|---:|---:|---:|---:|---:|
| `standard_steps100` | 142 | 114 | 87 | 67 | 14 | 13 | 0 |
| `experiment_a_steps200` | 141 | 112 | 89 | 63 | 13 | 13 | 0 |
| `experiment_b_calcnum` | 143 | 108 | 87 | 67 | 13 | 14 | 1 |
| `experiment_d_hardmix` | 144 | 106 | 86 | 63 | 12 | 12 | 2 |
| `experiment_f_steps250_r16a32` | 156 | 100 | 88 | 58 | 13 | 13 | 1 |

Experiment F improves exact accuracy and reduces date/axis and text/label errors, but it has the worst calculation error count and the lowest numeric relaxed accuracy. This explains why F is best exact but not best relaxed on the full validation set.

## Conclusion

Full val changes the earlier val100 interpretation:

- Best exact single adapter: `experiment_f_steps250_r16a32` at 69.48%.
- Best relaxed single adapter: `experiment_d_hardmix` at 77.86%.
- The spread among all trained adapters is small: 0.89 pp exact and 0.62 pp relaxed.
- The full-val result supports a conservative claim: 3B QLoRA gives modest but real gains among adapter variants, while small training variants are tightly clustered.
- Because no full-val baseline was run in module 19, this record cannot yet support a full-val baseline-vs-finetuned improvement claim.

## Recommended Next Steps

1. Run the 3B baseline on full val1920 with the same evaluator and output directory.
2. Optionally run `standard_numeric_final` on full val as a prompt-only control.
3. Compare full-val baseline vs `experiment_d_hardmix` and `experiment_f_steps250_r16a32`.
4. Build a full-val selector diagnostic using evaluated JSONL from the five adapters.
5. Audit a stratified subset of the 333 all-trained-adapters-wrong examples before making qualitative claims.
