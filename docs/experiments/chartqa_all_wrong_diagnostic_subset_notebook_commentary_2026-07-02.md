## 21.7 中文结果解读与下一步判断

### 本轮运行状态

Module 21 已完整跑完，`21.1 -> 21.6` 均成功返回，未发现 notebook cell 报错。

本轮只跑推荐的 all-wrong diagnostic subset，共 85 条；没有跑 full val，也没有训练新 LoRA。这个子集中的样本按定义都是原始 7 个 full-val run 全部 relaxed-wrong 的样本。

### 核心结果

| run | relaxed | exact | 备注 |
|---|---:|---:|---|
| `table_json_only` | 14/85 = 16.47% | 9/85 = 10.59% | 单 run 最好，结构化抽取方向有信号 |
| `hardmix_maxpix_602112` | 13/85 = 15.29% | 12/85 = 14.12% | image-only 最好之一 |
| `hardmix_maxpix_802816` | 13/85 = 15.29% | 12/85 = 14.12% | 与 602112 持平 |
| `image_plus_table_json` | 13/85 = 15.29% | 11/85 = 12.94% | 与 hardmix high-res 持平 |
| `f_maxpix_802816` | 12/85 = 14.12% | 12/85 = 14.12% | exact 不错，但 relaxed 不占优 |
| `baseline_maxpix_802816` | 10/85 = 11.76% | 9/85 = 10.59% | 纯高分辨率 baseline 有小幅追回 |
| `hardmix_axis_legend_prompt_802816` | 10/85 = 11.76% | 9/85 = 10.59% | 显式 axis/legend prompt 未带来提升 |

全部 Module 21 方法合并后的 oracle 为：

```text
26/85 = 30.59%
```

仍然全部错误：

```text
59/85 = 69.41%
```

chart-to-JSON extraction 共生成 85 条，其中 66 条能被解析为合法 JSON。

### 关键观察

1. **高分辨率有帮助，但不是充分解法。**

`baseline_maxpix_802816` 追回 10 条，说明部分错误确实和分辨率/可读性有关。但 hardmix 从 `602112` 提到 `802816` 后结果仍是 13/85，没有继续提升。

2. **hardmix 仍是 image-only 路线中更稳的 adapter。**

`hardmix_maxpix_602112` 和 `hardmix_maxpix_802816` 均为 13/85，高于 `f_maxpix_802816` 的 12/85，也高于 baseline 的 10/85。

3. **当前 axis/legend 显式 prompt 不值得作为主线。**

`hardmix_axis_legend_prompt_802816` 只有 10/85，低于普通 hardmix high-res。说明这个 prompt 没有稳定改善 grounding，后续如果继续做，应改成更细的分步抽取，而不是单句提示。

4. **table / structured extraction 是最值得继续的方向。**

`table_json_only` 是单 run 最好，并且有 5 个独有追回样本。这说明结构化读图能救回一部分 image-only 救不回的样本。

5. **但当前一次性 chart-to-JSON 还不够强。**

虽然 `table_json_only` 最好，但也只有 14/85。说明下一步不能只把这版 extraction 直接扩到 full val；应该先改进 extraction 质量，尤其是轴、图例、数据点、日期刻度和值的结构化还原。

### 按类别的 oracle 追回情况

| reviewed_primary | recovered / total |
|---|---:|
| `counting_or_category_count` | 5/8 |
| `text_label_lookup` | 7/16 |
| `yes_no_or_boolean` | 4/10 |
| `multi_step_calculation` | 3/9 |
| `numeric_value_or_scale` | 2/10 |
| `date_axis_reading` | 2/9 |
| `extreme_value_or_ranking` | 2/9 |
| `visual_mapping_or_legend` | 1/10 |
| `data_or_evaluator_issue` | 0/2 |
| `date_serial_or_label_format` | 0/2 |

比较容易被当前诊断追回的是 counting、text lookup、boolean/trend；仍然困难的是 date-axis、numeric scale、visual mapping。

### 当前判断

本轮结果进一步支持之前的判断：

> 不建议现在继续做新的 LoRA steps/rank 变体。下一步应继续围绕结构化读图、表格抽取、日期轴 grounding、图例颜色映射和 evaluator/data cleanup 展开。

推荐下一步：

```text
Module 22A: data/evaluator cleanup list
Module 22B: staged chart-to-table extraction on the same 85-sample subset
```

`Module 22B` 不应一次性要求模型输出完整 JSON，而应拆成：

1. chart type / title;
2. x-axis / y-axis / tick labels;
3. legend-color mapping;
4. visible text and data labels;
5. data_points table;
6. final QA 或 calculator-assisted QA。

这样能判断错误到底发生在 extraction 哪一步，而不是只看到最终 QA 仍然错误。

