# ChartQA Module 23C normalization v2 - 2026-07-03

## 运行口径

本模块只做 evaluator normalization v2：

- 不加载模型；
- 不使用 GPU；
- 不改 prediction；
- 不跑 full-val；
- 只读取 Module 23A 的 per-prediction 结果和 Module 23B 的诊断分组。

新增四类 normalization：

1. color synonyms and hex-to-color names；
2. trend morphology，例如 `increase / increases / increasing`；
3. numeric answer extraction from sentences when questions contain years；
4. list order plus singular/plural variants。

## 结果

| metric | before v2 | after v2 | gain |
|---|---:|---:|---:|
| oracle on clean-after-23A | 30/67 = 44.78% | 34/67 = 50.75% | +4 |

v2 oracle 追回样本：

```text
18, 241, 648, 816
```

按规则：

| rule | unique samples | indices |
|---|---:|---|
| `color_synonym_or_hex` | 2 | `18, 816` |
| `numeric_answer_in_sentence` | 1 | `241` |
| `trend_morphology` | 1 | `648` |

23B true-hard 口径变化：

```text
before v2: 28
after v2:  28
```

v2 后仍建议作为 targeted prompt ablation 的 true-hard 样本：

```text
28, 29, 189, 229, 250, 251, 281, 290, 291, 310, 312, 326, 344, 381, 408, 424, 467, 523, 529, 667, 675, 778, 781, 797, 804, 877, 901, 1055
```

## 解释

Normalization v2 主要用于剥离残留 evaluator 问题，避免 targeted prompt ablation 被颜色名、趋势词、句子数字抽取和列表格式问题污染。后续 prompt ablation 应默认使用 v2 后的 true-hard 样本。
