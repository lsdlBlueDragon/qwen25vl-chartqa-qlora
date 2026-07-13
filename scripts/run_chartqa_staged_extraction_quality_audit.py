#!/usr/bin/env python
"""Module 22C: local quality audit for staged extraction diagnostics.

This script does not load a model or use GPU. It reads existing Module 21,
22A, and 22B outputs, then produces per-sample audit tables and a Chinese
summary report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STAGED_DIR = Path("outputs/chartqa_all_wrong_diagnostics_from_drive/staged_extraction")
DEFAULT_M21_EVAL_DIR = Path("outputs/chartqa_all_wrong_diagnostics_from_drive/evaluated")
DEFAULT_SUBSET = Path("data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl")
DEFAULT_EXCLUDE_CSV = Path("outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/chartqa_staged_extraction_quality_audit_22c")
DEFAULT_REPORT = Path("docs/experiments/chartqa_staged_extraction_quality_audit_22c_2026-07-03.md")

M21_RUN_FILES = [
    "baseline_maxpix_802816_evaluated.jsonl",
    "f_maxpix_802816_evaluated.jsonl",
    "hardmix_axis_legend_prompt_802816_evaluated.jsonl",
    "hardmix_maxpix_602112_evaluated.jsonl",
    "hardmix_maxpix_802816_evaluated.jsonl",
    "image_plus_table_json_evaluated.jsonl",
    "table_json_only_evaluated.jsonl",
]

M22B_RUN_FILES = [
    "staged_table_json_only_evaluated.jsonl",
    "staged_image_plus_table_json_evaluated.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_exclude_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {int(row["sample_index"]) for row in csv.DictReader(f)}


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    return stripped


def parse_stage_json(text: str) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    try:
        return json.loads(strip_code_fence(text)), None
    except Exception as exc:  # noqa: BLE001 - report parse error as data.
        return None, str(exc)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def compact_number_text(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def stage_contains_reference(stage_text: str, reference: Any) -> bool:
    ref = compact_number_text(reference)
    haystack = compact_number_text(stage_text)
    if not ref:
        return False
    if ref in haystack:
        return True
    try:
        ref_num = float(ref.rstrip("%"))
    except ValueError:
        return False
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", haystack)]
    return any(abs(num - ref_num) <= max(0.02 * abs(ref_num), 0.05) for num in nums)


def count_list_field(obj: Any, key: str) -> int:
    if isinstance(obj, dict) and isinstance(obj.get(key), list):
        return len(obj[key])
    return 0


def bool_field(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value in (None, ""):
        return ""
    return str(value)


def load_run_correctness(eval_dir: Path, files: list[str]) -> dict[str, dict[int, dict[str, Any]]]:
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    for file_name in files:
        path = eval_dir / file_name
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        run_name = rows[0].get("run_name") or file_name.replace("_evaluated.jsonl", "")
        runs[run_name] = {int(row["sample_index"]): row for row in rows}
    return runs


def classify_failure(
    row: dict[str, Any],
    m21_oracle: bool,
    b22_oracle: bool,
    data_table_json_valid: bool,
    axes_json_valid: bool,
    table_contains_ref: bool,
    image_plus_correct: bool,
    table_only_correct: bool,
) -> str:
    if b22_oracle and m21_oracle:
        return "both_recovered"
    if b22_oracle and not m21_oracle:
        return "22b_unique_recovery"
    if m21_oracle and not b22_oracle:
        return "module21_unique_recovery"
    if not data_table_json_valid:
        return "stage_json_or_schema_failure"
    if not axes_json_valid:
        return "axis_legend_json_failure_but_table_parsed"
    if table_contains_ref and not image_plus_correct and not table_only_correct:
        return "table_may_contain_answer_but_qa_failed"
    category = row.get("reviewed_primary", "")
    if category in {"numeric_value_or_scale", "date_axis_reading", "visual_mapping_or_legend"}:
        return f"likely_visual_extraction_error:{category}"
    if category in {"multi_step_calculation", "extreme_value_or_ranking"}:
        return f"likely_reasoning_or_aggregation_error:{category}"
    return "still_wrong_needs_manual_visual_audit"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged-dir", type=Path, default=DEFAULT_STAGED_DIR)
    parser.add_argument("--module21-eval-dir", type=Path, default=DEFAULT_M21_EVAL_DIR)
    parser.add_argument("--subset-jsonl", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--exclude-csv", type=Path, default=DEFAULT_EXCLUDE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    subset_rows = {int(row["sample_index"]): row for row in read_jsonl(args.subset_jsonl)}
    excluded = load_exclude_indices(args.exclude_csv)
    valid_indices = sorted(idx for idx in subset_rows if idx not in excluded)

    stage_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for stage in ["overview", "axes_legend", "data_table"]:
        path = args.staged_dir / f"{stage}.jsonl"
        stage_rows[stage] = {int(row["sample_index"]): row for row in read_jsonl(path)}

    m21_runs = load_run_correctness(args.module21_eval_dir, M21_RUN_FILES)
    b22_runs = load_run_correctness(args.staged_dir, M22B_RUN_FILES)

    m21_recovered_by_idx: dict[int, list[str]] = defaultdict(list)
    b22_recovered_by_idx: dict[int, list[str]] = defaultdict(list)
    for run_name, rows in m21_runs.items():
        for idx, row in rows.items():
            if idx in valid_indices and row.get("eval_relaxed_correct"):
                m21_recovered_by_idx[idx].append(run_name)
    for run_name, rows in b22_runs.items():
        for idx, row in rows.items():
            if idx in valid_indices and row.get("eval_relaxed_correct"):
                b22_recovered_by_idx[idx].append(run_name)

    audit_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for idx in valid_indices:
        subset = subset_rows[idx]
        overview = stage_rows["overview"].get(idx, {})
        axes = stage_rows["axes_legend"].get(idx, {})
        table = stage_rows["data_table"].get(idx, {})

        parsed_overview, overview_error = parse_stage_json(overview.get("stage_output", ""))
        parsed_axes, axes_error = parse_stage_json(axes.get("stage_output", ""))
        parsed_table, table_error = parse_stage_json(table.get("stage_output", ""))

        for stage, err in [("overview", overview_error), ("axes_legend", axes_error), ("data_table", table_error)]:
            if err:
                parse_errors.append({"sample_index": idx, "stage": stage, "parse_error": err})

        table_text = table.get("stage_output", "")
        reference_answer = subset.get("reference_answer", "")
        table_contains_ref = stage_contains_reference(table_text, reference_answer)
        m21_oracle = bool(m21_recovered_by_idx.get(idx))
        b22_oracle = bool(b22_recovered_by_idx.get(idx))
        table_only_row = b22_runs.get("staged_table_json_only", {}).get(idx, {})
        image_plus_row = b22_runs.get("staged_image_plus_table_json", {}).get(idx, {})
        table_only_correct = bool(table_only_row.get("eval_relaxed_correct"))
        image_plus_correct = bool(image_plus_row.get("eval_relaxed_correct"))

        failure_class = classify_failure(
            subset,
            m21_oracle=m21_oracle,
            b22_oracle=b22_oracle,
            data_table_json_valid=table_error is None,
            axes_json_valid=axes_error is None,
            table_contains_ref=table_contains_ref,
            image_plus_correct=image_plus_correct,
            table_only_correct=table_only_correct,
        )

        audit_rows.append(
            {
                "sample_index": idx,
                "reviewed_primary": subset.get("reviewed_primary", ""),
                "question": subset.get("question", ""),
                "reference_answer": reference_answer,
                "module21_oracle": bool_field(m21_oracle),
                "module21_recovered_by": ";".join(m21_recovered_by_idx.get(idx, [])),
                "module22b_oracle": bool_field(b22_oracle),
                "module22b_recovered_by": ";".join(b22_recovered_by_idx.get(idx, [])),
                "recovery_relation": failure_class,
                "overview_json_valid": bool_field(overview_error is None),
                "axes_legend_json_valid": bool_field(axes_error is None),
                "data_table_json_valid": bool_field(table_error is None),
                "data_points_count": count_list_field(parsed_table, "data_points"),
                "candidate_values_count": count_list_field(parsed_table, "candidate_values_for_question"),
                "missing_uncertain_count": count_list_field(parsed_table, "missing_or_uncertain_values"),
                "arithmetic_needed": parsed_table.get("arithmetic_needed", "") if isinstance(parsed_table, dict) else "",
                "table_contains_reference_heuristic": bool_field(table_contains_ref),
                "staged_table_json_only_prediction": table_only_row.get("eval_prediction", ""),
                "staged_image_plus_table_json_prediction": image_plus_row.get("eval_prediction", ""),
                "issue_note": subset.get("issue_note", ""),
                "review_flags": subset.get("review_flags", ""),
                "image_path": subset.get("image_path", ""),
            }
        )

    by_relation = Counter(row["recovery_relation"] for row in audit_rows)
    by_category = Counter(row["reviewed_primary"] for row in audit_rows)
    category_relation: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit_rows:
        category_relation[row["reviewed_primary"]][row["recovery_relation"]] += 1

    m21_set = {idx for idx in valid_indices if m21_recovered_by_idx.get(idx)}
    b22_set = {idx for idx in valid_indices if b22_recovered_by_idx.get(idx)}
    combined_set = m21_set | b22_set
    still_wrong = sorted(set(valid_indices) - combined_set)

    high_value_review = [
        row
        for row in audit_rows
        if row["recovery_relation"]
        in {
            "22b_unique_recovery",
            "module21_unique_recovery",
            "table_may_contain_answer_but_qa_failed",
            "stage_json_or_schema_failure",
            "axis_legend_json_failure_but_table_parsed",
        }
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.output_dir / "chartqa_22c_staged_extraction_quality_audit.csv"
    audit_json = args.output_dir / "chartqa_22c_staged_extraction_quality_audit.json"
    review_csv = args.output_dir / "chartqa_22c_high_value_manual_review_queue.csv"
    summary_json = args.output_dir / "chartqa_22c_quality_audit_summary.json"

    fieldnames = [
        "sample_index",
        "reviewed_primary",
        "question",
        "reference_answer",
        "module21_oracle",
        "module21_recovered_by",
        "module22b_oracle",
        "module22b_recovered_by",
        "recovery_relation",
        "overview_json_valid",
        "axes_legend_json_valid",
        "data_table_json_valid",
        "data_points_count",
        "candidate_values_count",
        "missing_uncertain_count",
        "arithmetic_needed",
        "table_contains_reference_heuristic",
        "staged_table_json_only_prediction",
        "staged_image_plus_table_json_prediction",
        "issue_note",
        "review_flags",
        "image_path",
    ]
    write_csv(audit_csv, audit_rows, fieldnames)
    write_json(audit_json, audit_rows)
    write_csv(review_csv, high_value_review, fieldnames)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_or_model_used": False,
        "valid_subset_count": len(valid_indices),
        "excluded_reference_issue_count": len(excluded),
        "excluded_reference_issue_indices": sorted(excluded),
        "module21_oracle_count": len(m21_set),
        "module22b_oracle_count": len(b22_set),
        "combined_oracle_count": len(combined_set),
        "combined_oracle_accuracy": len(combined_set) / len(valid_indices),
        "still_wrong_count": len(still_wrong),
        "still_wrong_indices": still_wrong,
        "module22b_unique_indices": sorted(b22_set - m21_set),
        "module21_unique_indices": sorted(m21_set - b22_set),
        "both_recovered_indices": sorted(m21_set & b22_set),
        "parse_errors": parse_errors,
        "by_recovery_relation": dict(by_relation),
        "by_reviewed_primary": dict(by_category),
        "by_reviewed_primary_and_relation": {
            category: dict(counter) for category, counter in sorted(category_relation.items())
        },
        "outputs": {
            "audit_csv": str(audit_csv),
            "audit_json": str(audit_json),
            "high_value_manual_review_queue_csv": str(review_csv),
            "report_md": str(args.report_md),
        },
    }
    write_json(summary_json, summary)

    report = build_report(summary, audit_rows)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report, encoding="utf-8")
    (args.output_dir / "chartqa_22c_quality_audit_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def md_list(values: list[int], max_items: int = 80) -> str:
    if not values:
        return "`none`"
    shown = values[:max_items]
    suffix = "" if len(values) <= max_items else f" ... (+{len(values) - max_items})"
    return "`" + ", ".join(str(v) for v in shown) + suffix + "`"


def build_report(summary: dict[str, Any], audit_rows: list[dict[str, Any]]) -> str:
    relation_lines = "\n".join(
        f"| `{name}` | {count} |" for name, count in sorted(summary["by_recovery_relation"].items())
    )
    category_rows = []
    for category, counter in summary["by_reviewed_primary_and_relation"].items():
        total = sum(counter.values())
        combined = (
            counter.get("both_recovered", 0)
            + counter.get("22b_unique_recovery", 0)
            + counter.get("module21_unique_recovery", 0)
        )
        category_rows.append((category, total, combined, counter))
    category_lines = "\n".join(
        f"| `{cat}` | {total} | {combined} | {counter.get('22b_unique_recovery', 0)} | "
        f"{counter.get('module21_unique_recovery', 0)} | {counter.get('both_recovered', 0)} |"
        for cat, total, combined, counter in sorted(category_rows)
    )

    table_contains_answer_but_wrong = [
        int(row["sample_index"])
        for row in audit_rows
        if row["recovery_relation"] == "table_may_contain_answer_but_qa_failed"
    ]
    schema_failures = [
        int(row["sample_index"])
        for row in audit_rows
        if row["recovery_relation"] in {"stage_json_or_schema_failure", "axis_legend_json_failure_but_table_parsed"}
    ]

    return f"""# ChartQA Module 22C staged extraction quality audit - 2026-07-03

## 运行口径

本模块已按要求作为纯本地审计完成：没有加载模型，没有调用 GPU，没有跑 full-val，也没有训练 LoRA。

输入来自已经落盘的 Module 21 / 22A / 22B 结果镜像：

- Module 22B staged extraction：`outputs/chartqa_all_wrong_diagnostics_from_drive/staged_extraction`
- Module 21 evaluated runs：`outputs/chartqa_all_wrong_diagnostics_from_drive/evaluated`
- Module 22A reference/evaluator 排除表：`outputs/chartqa_evaluator_cleanup`
- diagnostic subset：`data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl`

22A 排除 8 个高优先级 reference/evaluator 问题样本后，本轮有效样本仍为：

```text
{summary["valid_subset_count"]}
```

## 总体结果

| item | count |
|---|---:|
| Module 21 oracle | {summary["module21_oracle_count"]}/{summary["valid_subset_count"]} |
| Module 22B oracle | {summary["module22b_oracle_count"]}/{summary["valid_subset_count"]} |
| Module 21 + 22B combined oracle | {summary["combined_oracle_count"]}/{summary["valid_subset_count"]} = {summary["combined_oracle_accuracy"]:.2%} |
| combined still wrong | {summary["still_wrong_count"]}/{summary["valid_subset_count"]} |

22B 独有追回：

```text
{", ".join(str(v) for v in summary["module22b_unique_indices"])}
```

Module 21 独有追回：

```text
{", ".join(str(v) for v in summary["module21_unique_indices"])}
```

两者都没追回：

```text
{", ".join(str(v) for v in summary["still_wrong_indices"])}
```

## 自动归因分层

| relation / suspected layer | count |
|---|---:|
{relation_lines}

说明：

- `22b_unique_recovery` 表示分阶段抽取相对 Module 21 有独立价值，适合看它到底补上了哪类视觉或语义线索。
- `module21_unique_recovery` 表示 22B 分阶段链路反而损失了信息，适合检查 staged table 是否遗漏或误改了原图线索。
- `table_may_contain_answer_but_qa_failed` 表示表格文本里启发式能找到 reference，但最终 QA 仍错，更像 QA 推理、格式化或答案选择失败。
- `stage_json_or_schema_failure` / `axis_legend_json_failure_but_table_parsed` 是格式或阶段 schema 层问题。
- `likely_visual_extraction_error:*` 与 `likely_reasoning_or_aggregation_error:*` 是按人工类别和输出状态做的初筛，仍需人工看图确认。

## 按人工主类别看恢复情况

| reviewed_primary | total | combined recovered | 22B unique | Module21 unique | both |
|---|---:|---:|---:|---:|---:|
{category_lines}

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
{", ".join(str(v) for v in schema_failures) if schema_failures else "none"}
```

table 疑似含 reference 但 QA 仍错样本：

```text
{", ".join(str(v) for v in table_contains_answer_but_wrong) if table_contains_answer_but_wrong else "none"}
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
"""


if __name__ == "__main__":
    raise SystemExit(main())
