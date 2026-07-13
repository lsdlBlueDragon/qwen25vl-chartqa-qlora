# ChartQA all-wrong diagnostic subset results - 2026-07-02

## 结论摘要

Module 21 已完成推荐 all-wrong diagnostic subset 的输入侧诊断。子集共有 85 条，全部来自 full-val 325 个 all-runs-wrong 样本，因此原始七个 full-val run 在这些样本上均为 relaxed-wrong。

本轮没有跑 full val，也没有训练新 LoRA。实验只验证：

- 更高分辨率是否能救回样本；
- hardmix 与 steps250/r16/a32 在 hard subset 上的差异；
- axis/legend 显式提示是否有效；
- chart-to-JSON / table-assisted QA 是否比 image-only 更有潜力。

核心结果：

- 单 run 最好：`table_json_only`，14/85 = 16.47% relaxed。
- image-only 最好：`hardmix_maxpix_602112` 和 `hardmix_maxpix_802816`，均为 13/85 = 15.29% relaxed。
- 全部 Module 21 run 的 oracle：26/85 = 30.59%。
- 仍全部错误：59/85 = 69.41%。
- chart-to-JSON extraction 输出 85 条，其中 66 条是可解析 JSON。

## Run 结果

| run | relaxed | exact | 说明 |
|---|---:|---:|---|
| `table_json_only` | 14/85 = 16.47% | 9/85 = 10.59% | 单 run 最好，说明结构化抽取方向有信号 |
| `hardmix_maxpix_602112` | 13/85 = 15.29% | 12/85 = 14.12% | image-only 最好之一 |
| `hardmix_maxpix_802816` | 13/85 = 15.29% | 12/85 = 14.12% | 与 602112 完全持平 |
| `image_plus_table_json` | 13/85 = 15.29% | 11/85 = 12.94% | 与 hardmix high-res 持平 |
| `f_maxpix_802816` | 12/85 = 14.12% | 12/85 = 14.12% | exact 高，但 relaxed 不占优 |
| `baseline_maxpix_802816` | 10/85 = 11.76% | 9/85 = 10.59% | 纯高分辨率 baseline 有小幅追回 |
| `hardmix_axis_legend_prompt_802816` | 10/85 = 11.76% | 9/85 = 10.59% | 显式 axis/legend prompt 没有提升 |

## Oracle / 合并收益

全部 Module 21 方法合并后，至少被一种方法追回的样本为 26 条：

```text
13, 24, 132, 158, 162, 169, 175, 190, 220, 233, 255, 269, 362,
368, 369, 397, 426, 623, 700, 800, 882, 974, 1114, 1327, 1351, 1810
```

仍全部错误的样本为 59 条。说明当前高分辨率和一次性 chart-to-JSON 诊断能证明方向有用，但还不是足够强的解决方案。

只有 sample 623 被所有 Module 21 方法同时追回。这说明不同诊断路径互补性明显，但单一路径还不稳定。

## 独有追回样本

| run | unique recovered |
|---|---:|
| `table_json_only` | 5 |
| `image_plus_table_json` | 1 |
| `f_maxpix_802816` | 1 |
| `baseline_maxpix_802816` | 0 |
| `hardmix_maxpix_602112` | 0 |
| `hardmix_maxpix_802816` | 0 |
| `hardmix_axis_legend_prompt_802816` | 0 |

`table_json_only` 的 5 个独有样本是最重要的新信号：结构化读图确实能救回一部分 image-only 方法救不回的样本。

## 按 reviewed_primary 的 oracle 追回

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

解释：

- counting / text lookup / boolean 这几类最容易被当前诊断救回。
- date-axis、numeric scale、visual mapping 仍很难。
- data/evaluator issue 不应继续作为模型能力目标，应进入下一阶段 cleanup。

## 按 review_flags 的 oracle 追回

| flag | recovered / total |
|---|---:|
| `needs_table_or_value_extraction` | 22/67 |
| `resolution_sensitive` | 21/66 |
| `needs_legend_color_mapping` | 15/44 |
| `needs_axis_date_grounding` | 7/27 |
| `boolean_or_trend` | 4/10 |
| `list_answer_format` | 1/7 |
| `calculation_after_extraction` | 0/6 |
| `date_serial_reference` | 0/2 |

解释：

- `needs_table_or_value_extraction` 和 `resolution_sensitive` 的追回最多，支持继续做结构化抽取。
- `calculation_after_extraction` 为 0/6，说明当前 extraction 还没有稳定抽对数值，不能直接进入 calculator-only 阶段。
- `needs_axis_date_grounding` 只有 7/27，日期轴定位仍是硬瓶颈。

## 技术判断

1. 高分辨率有帮助，但收益有限。

`baseline_maxpix_802816` 能追回 10/85，说明一部分错误确实受分辨率影响。但 hardmix 从 602112 到 802816 没有增加，说明不是简单加 max_pixels 就能继续提升。

2. hardmix 仍是 image-only 主线中更稳的选择。

`hardmix_maxpix_602112` 和 `hardmix_maxpix_802816` 都是 13/85，高于 F 的 12/85，也高于 baseline 的 10/85。

3. axis/legend prompt 没有兑现预期。

`hardmix_axis_legend_prompt_802816` 只有 10/85，比普通 hardmix high-res 低。这个 prompt 不应作为下一步主线，最多作为失败对照记录。

4. table/structured extraction 是最值得继续的方向，但需要改进 extraction 质量。

`table_json_only` 单 run 最高，并且有 5 个独有追回样本；但总准确率只有 16.47%，说明当前一次性 Qwen2.5-VL chart-to-JSON 还不够可靠。下一步应改进 extraction prompt、JSON schema、分步抽取，以及验证每个 extraction 是否真的抽对轴、图例和值。

5. 不建议现在继续训练 LoRA。

Module 21 的证据进一步支持：瓶颈不是“再训一个 adapter”，而是图表结构化读数、日期轴 grounding、视觉元素定位和 evaluator/data 清理。

## 下一步建议

建议下一模块不要 full-val，继续在 85 条 subset 上做更精细的结构化抽取：

1. 把 chart-to-JSON 拆成多步：
   - title / chart type
   - axis ticks
   - legend-color mapping
   - series / bars / data labels
   - final table
2. 对 extraction 本身做自动/人工打分，而不是只看最终 QA。
3. 对 `date_axis_reading` 单独做 x-axis tick extraction prompt。
4. 对 `visual_mapping_or_legend` 单独做 legend crop / color mapping prompt。
5. 对 `data_or_evaluator_issue` 和 `date_serial_reference` 建立 ignore/fix list。

简短路线：

```text
Module 22A: evaluator/data cleanup list
Module 22B: staged chart-to-table extraction on the same 85-sample subset
```

