# ChartQA Module 23A cleanup + normalization-only ablation - 2026-07-03

## 运行口径

Module 23A 已完成。它是纯本地后处理模块：

- 不加载模型；
- 不使用 GPU；
- 不改任何 prediction；
- 不跑 full-val；
- 只读取现有 Module 21 / 22B evaluated JSONL。

本模块分两步：

1. 应用 cleanup list：在 22A 原有 8 条 exclude 的基础上，加入 Codex 视觉复核标记的 10 条 reference/evaluator 问题样本。
2. 做 normalization-only 消融：在 22A 后的 77 条 subset 上，只改 answer normalization，观察能追回多少。

## Cleanup List

22A 原始排除：

```text
12, 14, 105, 158, 470, 918, 1351, 1561
```

23A 新增 Codex cleanup：

```text
138, 317, 362, 779, 882, 946, 977, 987, 1065, 1190
```

23A 扩展后 exclude 总数：

```text
18
```

因此 clean-after-23A denominator 为：

```text
67
```

另有两个样本只标记为 ambiguous/reference-sensitive，默认不加入 exclude：

```text
245, 832
```

## Normalization Rules

23A 新增 normalization 只覆盖以下规则：

- list answer format：例如 `Czech Republic, New Zealand` vs `[Czech Republic, New Zealand]`；
- star year：例如 `2028* -> 2028`；
- categorical answer contained in sentence：例如句子中明确包含 `orange`；
- percent / close numeric：例如 `65%` 或 `65` vs `65.3`。

没有加入会扩大语义边界的规则，例如把 reference 错误的样本强行判对。

## 77 条上的 normalization-only 消融

| metric | before | after | gain |
|---|---:|---:|---:|
| oracle on valid77 | 30/77 = 38.96% | 32/77 = 41.56% | +2 |

normalization-only 追回的 unique sample：

```text
154, 455
```

按规则分布：

| rule | unique samples | sample indices |
|---|---:|---|
| `categorical_answer_contained_in_sentence` | 5 | `154, 362, 426, 1114, 1327` |
| `exact_after_text_cleanup` | 3 | `362, 455, 882` |

## Clean Denominator 口径

应用 23A expanded cleanup 后：

| metric | before normalization | after normalization |
|---|---:|---:|
| oracle on clean-after-23A | 28/67 = 41.79% | 30/67 = 44.78% |

## Per-run Summary

| run | valid77 before | valid77 after | gain | clean before | clean after |
|---|---:|---:|---:|---:|---:|
| `baseline_maxpix_802816` | 9/77 | 11/77 | +2 | 9/67 | 9/67 |
| `f_maxpix_802816` | 11/77 | 11/77 | +0 | 9/67 | 9/67 |
| `hardmix_axis_legend_prompt_802816` | 9/77 | 10/77 | +1 | 9/67 | 9/67 |
| `hardmix_maxpix_602112` | 12/77 | 13/77 | +1 | 11/67 | 11/67 |
| `hardmix_maxpix_802816` | 12/77 | 13/77 | +1 | 11/67 | 11/67 |
| `image_plus_table_json` | 12/77 | 14/77 | +2 | 12/67 | 13/67 |
| `table_json_only` | 13/77 | 16/77 | +3 | 13/67 | 15/67 |
| `staged_table_json_only` | 15/77 | 19/77 | +4 | 15/67 | 17/67 |
| `staged_image_plus_table_json` | 16/77 | 19/77 | +3 | 16/67 | 17/67 |

## 当前判断

23A 把两件事分开了：

- normalization-only 的收益说明有多少“答案已经基本对，但 evaluator 没吃到”；
- expanded cleanup 的收益说明当前 subset 里有多少不适合作为模型失败统计的 reference/evaluator 问题。

如果后续要做硬失败定向诊断，建议使用 clean-after-23A denominator，同时保留 ambiguous 样本单独统计。这样下一轮 strict threshold、date-axis、range aggregation、spatial grounding 等诊断不会被 reference cleanup 和 answer formatting 干扰。

## 输出文件

- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_expanded_cleanup_exclude_list.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_recovered_predictions.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_run_summary.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_summary.json`
- `docs/experiments/chartqa_23a_cleanup_normalization_2026-07-03.md`
