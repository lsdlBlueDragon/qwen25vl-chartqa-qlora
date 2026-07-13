# ChartQA Module 23B hard failure targeted diagnostics - 2026-07-03

## 运行口径

Module 23B 是定向诊断准备与初筛模块：

- 不加载模型；
- 不使用 GPU；
- 不改 prediction；
- 不跑 full-val；
- 基于 Module 23A 的 clean-after-23A + normalization 后口径。

23A clean denominator 为 67，normalization 后 oracle 追回 30，因此本轮硬失败样本数为：

```text
37
```

硬失败 index：

```text
18, 28, 29, 46, 189, 229, 241, 250, 251, 281, 290, 291, 310, 312, 326, 344, 381, 408, 424, 467, 523, 529, 571, 648, 667, 675, 688, 778, 781, 797, 804, 816, 877, 901, 976, 978, 1055
```

## 按人工主类别分布

| reviewed_primary | count |
|---|---:|
| `counting_or_category_count` | 2 |
| `date_axis_reading` | 3 |
| `extreme_value_or_ranking` | 5 |
| `multi_step_calculation` | 6 |
| `numeric_value_or_scale` | 6 |
| `text_label_lookup` | 6 |
| `visual_mapping_or_legend` | 6 |
| `yes_no_or_boolean` | 3 |

## 按定向诊断桶分布

| target_failure_bucket | count |
|---|---:|
| `arithmetic_average_or_median` | 2 |
| `arithmetic_sum_or_difference` | 3 |
| `boolean_after_computation_or_trend` | 3 |
| `date_axis_peak_or_extreme` | 3 |
| `extreme_or_ranking` | 2 |
| `label_after_computation` | 4 |
| `legend_color_mapping` | 6 |
| `multi_step_reasoning` | 1 |
| `numeric_value_extraction` | 1 |
| `range_or_threshold_aggregation` | 2 |
| `ranking_after_difference` | 1 |
| `semantic_category_filtering` | 1 |
| `spatial_position_grounding` | 2 |
| `specific_value_lookup_with_axis` | 3 |
| `text_label_lookup` | 2 |
| `timepoint_count_or_threshold_count` | 1 |

## 图像复核包

已生成按 target bucket 分组的 contact sheets：

- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\arithmetic_average_or_median.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\arithmetic_sum_or_difference.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\boolean_after_computation_or_trend.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\date_axis_peak_or_extreme.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\extreme_or_ranking.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\label_after_computation.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\legend_color_mapping.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\multi_step_reasoning.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\numeric_value_extraction.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\range_or_threshold_aggregation.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\ranking_after_difference.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\semantic_category_filtering.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\spatial_position_grounding.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\specific_value_lookup_with_axis.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\text_label_lookup.png`
- `outputs\chartqa_23b_hard_failure_diagnostics\contact_sheets\timepoint_count_or_threshold_count.png`

## Hard Failure Queue

| sample | target bucket | primary | question | reference |
|---:|---|---|---|---|
| 18 | `legend_color_mapping` | `visual_mapping_or_legend` | What is the colour of oppose in the graph? | Light blue |
| 28 | `legend_color_mapping` | `visual_mapping_or_legend` | What segment represent by dark grey color? | Both |
| 29 | `multi_step_reasoning` | `multi_step_calculation` | What is the percentage of both and don't know? | [4, 9] |
| 46 | `specific_value_lookup_with_axis` | `numeric_value_or_scale` | What is the death rate in the age group 5-14 years old? | 0.0034 |
| 189 | `semantic_category_filtering` | `counting_or_category_count` | How many age groups are mentioned in the given graph? | 4 |
| 229 | `date_axis_peak_or_extreme` | `date_axis_reading` | Which year recorded the highest number of cases of killing of male Journalists? | 2018 |
| 241 | `arithmetic_sum_or_difference` | `multi_step_calculation` | What is the approximate difference of values in the year 1990? | 900 |
| 250 | `date_axis_peak_or_extreme` | `date_axis_reading` | In which year, the value of Employment in the agriculture graph peaked? | 1999 |
| 251 | `timepoint_count_or_threshold_count` | `counting_or_category_count` | For how many years, the value of the "Employment in services" graph smaller than 60%? | 13 |
| 281 | `boolean_after_computation_or_trend` | `yes_no_or_boolean` | Is the sum of the highest value of navy blue bar and median of light blue bar greater than 100? | No |
| 290 | `legend_color_mapping` | `visual_mapping_or_legend` | What's the maximum value in the brightest yellow bar? | 53 |
| 291 | `arithmetic_sum_or_difference` | `multi_step_calculation` | What's the difference in the value of the total number of persons who want to improve the way government works and who have not? | 33 |
| 310 | `extreme_or_ranking` | `extreme_value_or_ranking` | What's the value of the 1st Longest bar in the graph? | 25 |
| 312 | `legend_color_mapping` | `visual_mapping_or_legend` | What does the Dark blue bar represent? | A great deal |
| 326 | `spatial_position_grounding` | `extreme_value_or_ranking` | What's the value of the rightmost bar in the middle? | 50 |
| 344 | `spatial_position_grounding` | `extreme_value_or_ranking` | What's the value of the rightmost first bar from the bottom? | 8 |
| 381 | `boolean_after_computation_or_trend` | `yes_no_or_boolean` | Is the difference of value of Austria and Ireland bar is greater then the value of United States bar? | No |
| 408 | `specific_value_lookup_with_axis` | `numeric_value_or_scale` | What is Republican data in Feb 2015 for mostly good? | [12,63,23] |
| 424 | `legend_color_mapping` | `visual_mapping_or_legend` | Which price is represented by brown color bar? | Northwest Europe marker price |
| 467 | `arithmetic_average_or_median` | `multi_step_calculation` | Take highest percentage and lowest percentage (leave 0), add it and divide it by 2, what is the result? | 17.5 |
| 523 | `arithmetic_average_or_median` | `multi_step_calculation` | What is the average price for snowboard boots? (in dolalrs)? | 158.58 |
| 529 | `label_after_computation` | `text_label_lookup` | Which site is three times than yellowpages? | Industry specific |
| 571 | `arithmetic_sum_or_difference` | `multi_step_calculation` | What's the total number of  deaths caused by major droughts worldwide in Sudan at 1983? | 250000 |
| 648 | `boolean_after_computation_or_trend` | `yes_no_or_boolean` | Does the life expectancy increase or decrease over time? | increasing |
| 667 | `label_after_computation` | `text_label_lookup` | What 2 slices make up over 75% of the crowdfunding total? | [Equity-based crowdfunding, Real estate crowdfunding] |
| 675 | `range_or_threshold_aggregation` | `numeric_value_or_scale` | What's the percentage value of purchases by people over 55 years old? | 22 |
| 688 | `label_after_computation` | `text_label_lookup` | Which two market shares have been taken? | [Investment funds, Discretionary mandate assets] |
| 778 | `text_label_lookup` | `text_label_lookup` | Which age group is "very likely" to subscribe to Disney's new online video streaming service? | 30-44 |
| 781 | `label_after_computation` | `text_label_lookup` | For which of the degrees is there the biggest gap between the median weekly earnings of full-time wage and salary workers in 2020 in the US? | Some college or associate's degree |
| 797 | `date_axis_peak_or_extreme` | `date_axis_reading` | Which year(s) had the greatest difference between the soft drink price and hot dog price? | [2010/11, 2011/12, 2012/13] |
| 804 | `extreme_or_ranking` | `extreme_value_or_ranking` | what is the highest value in blue bar ? | 47 |
| 816 | `legend_color_mapping` | `visual_mapping_or_legend` | Which color does men indicate in the graph? | light blue |
| 877 | `ranking_after_difference` | `extreme_value_or_ranking` | For which social network, the percentage is minimum between male and female? | Twitter |
| 901 | `range_or_threshold_aggregation` | `numeric_value_or_scale` | How much is the e commerce sales for companies under 250 employees in 2019? | 83.4 |
| 976 | `text_label_lookup` | `text_label_lookup` | What is included in the statistics for people originating from Greenland moving back home to Greenland as well as Danish people born in Denmark moving to | Danish citizenship |
| 978 | `specific_value_lookup_with_axis` | `numeric_value_or_scale` | What was the revenue of the software and ICT solutions sector in 2019? | 12.3 |
| 1055 | `numeric_value_extraction` | `numeric_value_or_scale` | What was the market share of discretionary mandate assets in Europe at the end of 2018? | 45.4 |

## 当前用途

这份队列用于后续人工视觉复核和 prompt/evaluator ablation 设计。23B 的下一步不是直接训练，而是逐类确认：

- 哪些是视觉读数/颜色映射失败；
- 哪些是空间语言或类别过滤失败；
- 哪些是先计算再选标签的推理失败；
- 哪些仍有残留 reference/question ambiguity。
