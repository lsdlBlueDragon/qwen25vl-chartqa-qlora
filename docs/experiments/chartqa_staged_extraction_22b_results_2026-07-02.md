# ChartQA Module 22B staged extraction results - 2026-07-02

## 运行状态

Module 22B 已完整运行完成：

- `22B.1` 成功恢复 subset、exclude list 和 helper script；
- `22B.2` 成功完成 staged extraction 和两种 QA；
- `22B.3` 成功读取 summary；
- notebook cell 无报错，`returncode=0`。

本模块没有跑 full-val，也没有训练新 LoRA。

## 数据口径

原始 recommended diagnostic subset 有 85 条。

Module 22A 标出的 8 条 `exclude_or_fix_reference` 样本被默认跳过：

```text
12, 14, 105, 158, 470, 918, 1351, 1561
```

所以 Module 22B 的有效诊断样本数为：

```text
77
```

## Staged extraction JSON 合法率

| stage | valid JSON |
|---|---:|
| `overview` | 77/77 |
| `axes_legend` | 75/77 |
| `data_table` | 74/77 |

无效 JSON 样本：

```text
axes_legend: 1190, 1114
data_table: 13, 1327, 233
```

解释：

- 分步 schema 明显提升了 JSON 可解析率。
- Module 21 one-shot chart-to-JSON 是 66/85；22B 的关键 `data_table` 阶段是 74/77。
- 但 JSON 合法并不等于数值抽取正确。后续要看 QA 是否提升。

## 22B QA 结果

| run | relaxed |
|---|---:|
| `staged_table_json_only` | 15/77 = 19.48% |
| `staged_image_plus_table_json` | 16/77 = 20.78% |
| 22B oracle | 23/77 = 29.87% |

22B 追回样本：

```text
132, 162, 169, 175, 186, 220, 233, 245, 269, 368, 397, 419,
426, 441, 623, 677, 700, 800, 832, 974, 1114, 1327, 1810
```

仍错误样本数：

```text
54/77
```

## 与 Module 21 对比

使用同样的 77 条有效样本口径：

| method group | oracle |
|---|---:|
| Module 21 high-res / one-shot table diagnostics | 24/77 = 31.17% |
| Module 22B staged extraction diagnostics | 23/77 = 29.87% |
| Module 21 + Module 22B combined oracle | 30/77 = 38.96% |

结论：

- 22B 单独没有超过 Module 21 的整体 oracle。
- 但 22B 追回了 6 个 Module 21 没追回的新样本。
- 组合后从 24/77 提升到 30/77，说明 staged extraction 有互补性，但还不是更强的单一路线。

## 22B 独有追回样本

22B 相对 Module 21 独有追回：

```text
186, 245, 419, 441, 677, 832
```

类别分布：

| reviewed_primary | count |
|---|---:|
| `date_axis_reading` | 2 |
| `counting_or_category_count` | 1 |
| `text_label_lookup` | 1 |
| `visual_mapping_or_legend` | 1 |
| `yes_no_or_boolean` | 1 |

解释：

- staged extraction 对部分 date-axis 和 visual/text/boolean 样本确实有新增价值。
- 但新增样本不集中在 numeric/calculation 主瓶颈上。

## Module 21 独有、22B 没追回的样本

Module 21 相对 22B 独有追回：

```text
13, 24, 190, 255, 362, 369, 882
```

类别分布：

| reviewed_primary | count |
|---|---:|
| `counting_or_category_count` | 2 |
| `date_axis_reading` | 1 |
| `multi_step_calculation` | 1 |
| `numeric_value_or_scale` | 1 |
| `text_label_lookup` | 1 |
| `yes_no_or_boolean` | 1 |

解释：

- staged extraction 在一些原本 one-shot/table 或 high-res 能解决的样本上发生回退。
- 这说明分步过程虽然更可解析，但可能在 stage 间传递时丢信息，或者 data_table 阶段没有保留足够数值。

## Combined oracle by category

Module 21 + 22B 合并后，按类别追回：

| reviewed_primary | recovered / total |
|---|---:|
| `counting_or_category_count` | 6/8 |
| `yes_no_or_boolean` | 5/9 |
| `text_label_lookup` | 7/15 |
| `date_axis_reading` | 4/9 |
| `multi_step_calculation` | 3/9 |
| `visual_mapping_or_legend` | 2/9 |
| `numeric_value_or_scale` | 2/10 |
| `extreme_value_or_ranking` | 1/8 |

仍然最难的是：

- numeric value / scale;
- extreme/ranking;
- visual mapping / legend;
- multi-step calculation。

## 当前技术判断

1. 分步抽取提高了 JSON 格式稳定性。

`overview`、`axes_legend`、`data_table` 的合法 JSON 率都高于 one-shot extraction。这说明 staged schema 是更可控的工程方向。

2. 但分步抽取没有带来足够的 QA 增益。

22B oracle 是 23/77，略低于 Module 21 的 24/77。说明问题不只是 JSON 格式，而是抽取内容的正确性，尤其是 data values、axis/date alignment、legend mapping 是否真的抽对。

3. image + staged table 略好于 table-only。

`staged_image_plus_table_json` 是 16/77，高于 `staged_table_json_only` 的 15/77。这说明 extraction 还不完整，模型仍需要回看图像。

4. 22B 有互补性，但不能直接替代 Module 21。

组合 oracle 提升到 30/77，说明 staged extraction 值得保留为诊断路线，但不能说它已经是更强 pipeline。

5. 现在仍不建议继续训练 LoRA 或扩 full-val。

剩余 47/77 连 Module 21 + 22B 都没救回。继续 full-val 只会放大成本，不能解释失败原因。

## 下一步建议

下一步不应直接扩展 22B，而应先做 22C：抽取质量审计。

建议：

```text
Module 22C: staged extraction quality audit
```

目标：

1. 对 77 条的 `overview / axes_legend / data_table` 做抽取质量打分；
2. 标出失败发生在哪个阶段：
   - axis/tick/date 错；
   - legend/color mapping 错；
   - data point 缺失；
   - 数值尺度错；
   - table 对了但 QA 算错；
3. 对 22B 独有追回和 Module 21 独有追回分别抽样看原因；
4. 再决定是否做 crop-based extraction。

推荐 22C 输出：

```text
staged_extraction_quality_audit.csv
staged_extraction_quality_audit_summary.json
```

目前最有价值的问题不是“能不能多跑一点”，而是：

> 为什么 JSON 合法率很高，但 QA 只追回 20% 左右？

这个问题答清楚之后，才值得进入 crop、OCR、或更细 schema。

