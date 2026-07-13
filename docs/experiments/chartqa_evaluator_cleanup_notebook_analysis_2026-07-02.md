## 22A.5 最新运行结果分析

### 运行状态

Module 22A 已成功运行完成：

- `22A.1` 成功从 Drive 恢复输入和 helper 脚本；
- `22A.2` 成功生成 evaluator/data cleanup candidates；
- `22A.3` 成功读取 summary 和 markdown report；
- 未发现 notebook cell 报错。

本模块没有跑模型，没有跑 full-val，也没有修改历史指标。

### 22A 清理结果

推荐 85 条 diagnostic subset 中，识别出：

```text
cleanup candidates: 21/85
exclude_or_fix_reference: 8
normalization_candidate: 6
answer_format_manual_review: 7
```

如果只把最高优先级的 `exclude_or_fix_reference` 从模型能力判断中剥离，则有效模型错误样本数从 85 变为：

```text
77
```

### issue type 分布

| issue_type | count |
|---|---:|
| `answer_type_or_reference_mismatch` | 4 |
| `date_serial_reference` | 2 |
| `general_data_or_evaluator_issue` | 2 |
| `list_answer_format` | 7 |
| `ocr_spelling_near_miss` | 3 |
| `color_granularity` | 1 |
| `scale_normalization` | 1 |
| `semantic_equivalence` | 1 |

### 高优先级剔除/修正样本

这些样本不应直接用于判断模型能力，应先修 reference 或单独统计：

```text
12, 14, 105, 158, 470, 918, 1351, 1561
```

原因包括：

- Excel serial date reference；
- 问题问数值但 reference 是 label；
- 问题问 label 但 reference 是数值；
- 明显 annotation/reference mismatch。

### normalization / 格式复核样本

normalization candidates：

```text
18, 46, 648, 1114, 1327, 1810
```

典型问题：

- `Light blue` vs `Blue`
- `0.0034` vs `0.34`
- `increasing` vs `increase`
- OCR spelling near miss

answer-format manual review：

```text
29, 362, 408, 455, 667, 688, 797
```

典型问题：

- list-vs-sum；
- list 顺序；
- list 部分正确；
- 题目到底要求单值还是多个值不够清晰。

### 对 Module 21 结论的影响

Module 21 原始 oracle：

```text
26/85 = 30.59%
```

剔除 8 个 `exclude_or_fix_reference` 后，有效 denominator 为 77。Module 21 oracle 追回样本中有 2 个属于 `exclude_or_fix_reference`：

```text
158, 1351
```

因此有效口径下的 Module 21 oracle 为：

```text
24/77 = 31.17%
```

这个变化很小，说明 Module 21 的主要结论没有被 evaluator/data issue 推翻：

> 高分辨率和一次性 chart-to-JSON 有一定帮助，但仍不足以解决大多数 hard subset；下一步仍应做 staged chart-to-table extraction，而不是继续 LoRA rank/steps。

### 当前判断

22A 的价值是把“模型真的不会”和“评测/标注/格式口径不干净”拆开了。

下一步建议仍然是：

```text
Module 22B: staged chart-to-table extraction on the same 85-sample subset
```

22B 应优先避开或单独标记 8 个 `exclude_or_fix_reference` 样本，防止错误 reference 污染 extraction 诊断。

