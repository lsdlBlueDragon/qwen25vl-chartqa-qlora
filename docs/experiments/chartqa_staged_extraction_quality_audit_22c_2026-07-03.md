# ChartQA Module 22C staged extraction quality audit - 2026-07-03

## 运行口径

本模块已按要求作为纯本地审计完成：没有加载模型，没有调用 GPU，没有跑 full-val，也没有训练 LoRA。

输入来自已经落盘的 Module 21 / 22A / 22B 结果镜像：

- Module 22B staged extraction：`outputs/chartqa_all_wrong_diagnostics_from_drive/staged_extraction`
- Module 21 evaluated runs：`outputs/chartqa_all_wrong_diagnostics_from_drive/evaluated`
- Module 22A reference/evaluator 排除表：`outputs/chartqa_evaluator_cleanup`
- diagnostic subset：`data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl`

22A 排除 8 个高优先级 reference/evaluator 问题样本后，本轮有效样本仍为：

```text
77
```

## 总体结果

| item | count |
|---|---:|
| Module 21 oracle | 24/77 |
| Module 22B oracle | 23/77 |
| Module 21 + 22B combined oracle | 30/77 = 38.96% |
| combined still wrong | 47/77 |

22B 独有追回：

```text
186, 245, 419, 441, 677, 832
```

Module 21 独有追回：

```text
13, 24, 190, 255, 362, 369, 882
```

两者都没追回：

```text
18, 28, 29, 46, 138, 154, 189, 229, 241, 250, 251, 281, 290, 291, 310, 312, 317, 326, 344, 381, 408, 424, 455, 467, 523, 529, 571, 648, 667, 675, 688, 778, 779, 781, 797, 804, 816, 877, 901, 946, 976, 977, 978, 987, 1055, 1065, 1190
```

## 自动归因分层

| relation / suspected layer | count |
|---|---:|
| `22b_unique_recovery` | 6 |
| `axis_legend_json_failure_but_table_parsed` | 1 |
| `both_recovered` | 17 |
| `likely_reasoning_or_aggregation_error:extreme_value_or_ranking` | 3 |
| `likely_reasoning_or_aggregation_error:multi_step_calculation` | 6 |
| `likely_visual_extraction_error:date_axis_reading` | 1 |
| `likely_visual_extraction_error:numeric_value_or_scale` | 5 |
| `likely_visual_extraction_error:visual_mapping_or_legend` | 4 |
| `module21_unique_recovery` | 7 |
| `still_wrong_needs_manual_visual_audit` | 7 |
| `table_may_contain_answer_but_qa_failed` | 20 |

说明：

- `22b_unique_recovery` 表示分阶段抽取相对 Module 21 有独立价值，适合看它到底补上了哪类视觉或语义线索。
- `module21_unique_recovery` 表示 22B 分阶段链路反而损失了信息，适合检查 staged table 是否遗漏或误改了原图线索。
- `table_may_contain_answer_but_qa_failed` 表示表格文本里启发式能找到 reference，但最终 QA 仍错，更像 QA 推理、格式化或答案选择失败。
- `stage_json_or_schema_failure` / `axis_legend_json_failure_but_table_parsed` 是格式或阶段 schema 层问题。
- `likely_visual_extraction_error:*` 与 `likely_reasoning_or_aggregation_error:*` 是按人工类别和输出状态做的初筛，仍需人工看图确认。

## 按人工主类别看恢复情况

| reviewed_primary | total | combined recovered | 22B unique | Module21 unique | both |
|---|---:|---:|---:|---:|---:|
| `counting_or_category_count` | 8 | 6 | 1 | 2 | 3 |
| `date_axis_reading` | 9 | 4 | 2 | 1 | 1 |
| `extreme_value_or_ranking` | 8 | 1 | 0 | 0 | 1 |
| `multi_step_calculation` | 9 | 3 | 0 | 1 | 2 |
| `numeric_value_or_scale` | 10 | 2 | 0 | 1 | 1 |
| `text_label_lookup` | 15 | 7 | 1 | 1 | 5 |
| `visual_mapping_or_legend` | 9 | 2 | 1 | 0 | 1 |
| `yes_no_or_boolean` | 9 | 5 | 1 | 1 | 3 |

## 高价值人工复核队列

已生成：

```text
outputs/chartqa_staged_extraction_quality_audit_22c/chartqa_22c_high_value_manual_review_queue.csv
```

优先看三类：

1. 22B 独有追回样本：确认 staged extraction 捕捉到了哪些 Module21 没捕捉到的线索。
2. Module21 独有追回样本：确认 22B 的 staged table 是否丢失信息或引入错误。
3. table 里疑似含 reference 但 QA 仍错的样本：判断下一步是否需要改 QA prompt/答案规范化，而不是继续改视觉抽取。

schema 或 JSON 层异常样本：

```text
1190
```

table 疑似含 reference 但 QA 仍错样本：

```text
28, 138, 154, 189, 229, 250, 281, 317, 326, 424, 529, 675, 778, 779, 781, 877, 946, 977, 987, 1065
```

## 当前判断

22C 的结论延续 22B：分阶段抽取让输出更可控，但不是单独更强的主线。它的价值在于提供互补样本和可审计中间态。

现在最值得继续的是小规模人工复核高价值队列，而不是马上 full-val 或继续 LoRA。复核目标不是重新判断总分，而是判断失败层级：

- 若 table 已有正确值但 QA 错，下一步优先做 QA prompt / answer normalization。
- 若 table 没有正确值，下一步优先做视觉读数、轴刻度、legend/color mapping 的专项提示或裁剪策略。
- 若 22B 相比 Module21 丢失正确样本，说明 staged extraction 有信息压缩损失，不能直接替代 image-only / one-shot 路线。

## 输出文件

- `outputs/chartqa_staged_extraction_quality_audit_22c/chartqa_22c_staged_extraction_quality_audit.csv`
- `outputs/chartqa_staged_extraction_quality_audit_22c/chartqa_22c_staged_extraction_quality_audit.json`
- `outputs/chartqa_staged_extraction_quality_audit_22c/chartqa_22c_high_value_manual_review_queue.csv`
- `outputs/chartqa_staged_extraction_quality_audit_22c/chartqa_22c_quality_audit_summary.json`
- `docs/experiments/chartqa_staged_extraction_quality_audit_22c_2026-07-03.md`
