#!/usr/bin/env python
"""Module 23B: targeted diagnostics for clean hard failures.

This is a file-only diagnostic helper. It reads Module 23A outputs, identifies
samples that remain wrong after cleanup + answer normalization, assigns a
targeted failure bucket, and optionally builds contact sheets for visual review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import shorten
from typing import Any


DEFAULT_SUBSET_JSONL = Path("data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl")
DEFAULT_23A_PER_PREDICTION = Path(
    "outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/chartqa_23b_hard_failure_diagnostics")
DEFAULT_REPORT_MD = Path("docs/experiments/chartqa_23b_hard_failure_diagnostics_2026-07-03.md")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def infer_target(question: str, category: str) -> tuple[str, str]:
    q = question.lower()
    if category == "visual_mapping_or_legend":
        if any(word in q for word in ["colour", "color", "dark", "light", "blue", "yellow", "brown"]):
            return "legend_color_mapping", "Question explicitly asks for color or color-coded series."
        return "legend_series_binding", "Question depends on mapping a visual encoding to a label."
    if category == "date_axis_reading":
        if any(word in q for word in ["highest", "peak", "greatest", "maximum"]):
            return "date_axis_peak_or_extreme", "Question asks for the year of an extreme value."
        return "date_axis_local_comparison", "Question depends on year/date grounding."
    if category == "counting_or_category_count":
        if "age group" in q:
            return "semantic_category_filtering", "Need count true age groups while excluding aggregate/statistical series."
        if "years" in q:
            return "timepoint_count_or_threshold_count", "Need count years satisfying a condition."
        return "category_counting", "Need count chart categories."
    if category == "multi_step_calculation":
        if any(word in q for word in ["average", "divide", "median"]):
            return "arithmetic_average_or_median", "Need aggregate numbers before answering."
        if any(word in q for word in ["difference", "sum", "total", "add"]):
            return "arithmetic_sum_or_difference", "Need compute sum/difference from chart values."
        return "multi_step_reasoning", "Need multi-hop chart reasoning."
    if category == "numeric_value_or_scale":
        if any(word in q for word in ["over 55", "under", "above", "less than", "smaller", "greater"]):
            return "range_or_threshold_aggregation", "Need combine ranges or apply threshold semantics."
        if any(word in q for word in ["2019", "2020", "feb", "year"]):
            return "specific_value_lookup_with_axis", "Need locate one value by series/date/category."
        return "numeric_value_extraction", "Need read a numeric value or scale."
    if category == "text_label_lookup":
        if any(word in q for word in ["gap", "difference", "three times", "over 75", "two"]):
            return "label_after_computation", "Need compute first, then map result back to a label."
        return "text_label_lookup", "Need extract or select a textual label."
    if category == "extreme_value_or_ranking":
        if any(word in q for word in ["rightmost", "bottom", "middle", "first"]):
            return "spatial_position_grounding", "Need map positional language to the correct bar/segment."
        if "between" in q or "difference" in q:
            return "ranking_after_difference", "Need rank differences rather than raw values."
        return "extreme_or_ranking", "Need identify an extreme value."
    if category == "yes_no_or_boolean":
        if any(word in q for word in ["sum", "difference", "greater", "increase", "decrease"]):
            return "boolean_after_computation_or_trend", "Need compute or infer trend before yes/no."
        return "boolean_question", "Need answer yes/no from chart."
    return "other_hard_failure", "Fallback bucket."


def build_contact_sheets(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return []

    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["target_failure_bucket"]].append(row)

    font = ImageFont.load_default()
    sheet_paths: list[str] = []
    for bucket, bucket_rows in sorted(grouped.items()):
        thumb_w, thumb_h = 320, 230
        label_h = 84
        cols = 2
        rows_n = (len(bucket_rows) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows_n * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(bucket_rows):
            x = (i % cols) * thumb_w
            y = (i // cols) * (thumb_h + label_h)
            image_path = Path(row["image_path"])
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
                sheet.paste(image, (x + (thumb_w - image.width) // 2, y))
            except Exception as exc:  # noqa: BLE001
                draw.text((x + 8, y + 8), f"image error: {exc}", fill="red", font=font)
            label = (
                f"{row['sample_index']} | {row['reviewed_primary']}\n"
                f"Q: {shorten(row['question'], width=70)}\n"
                f"Ref: {shorten(row['reference_answer'], width=45)}"
            )
            draw.rectangle((x, y + thumb_h, x + thumb_w, y + thumb_h + label_h), fill="#f7f7f7")
            draw.text((x + 6, y + thumb_h + 6), label, fill="black", font=font)
        safe_bucket = re.sub(r"[^a-zA-Z0-9_.-]+", "_", bucket)
        path = sheet_dir / f"{safe_bucket}.png"
        sheet.save(path)
        sheet_paths.append(str(path))
    return sheet_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-jsonl", type=Path, default=DEFAULT_SUBSET_JSONL)
    parser.add_argument("--per-prediction-csv", type=Path, default=DEFAULT_23A_PER_PREDICTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    subset = {int(row["sample_index"]): row for row in read_jsonl(args.subset_jsonl)}
    predictions = read_csv(args.per_prediction_csv)

    clean_indices: set[int] = set()
    normalized_oracle: set[int] = set()
    by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        idx = int(row["sample_index"])
        if not norm_bool(row["in_23a_cleanup_exclude"]):
            clean_indices.add(idx)
            by_idx[idx].append(row)
            if norm_bool(row["normalized_correct"]):
                normalized_oracle.add(idx)

    hard_indices = sorted(clean_indices - normalized_oracle)
    hard_rows: list[dict[str, Any]] = []
    for idx in hard_indices:
        meta = subset[idx]
        category = meta.get("reviewed_primary", "")
        target, rationale = infer_target(meta.get("question", ""), category)
        preds = []
        for row in by_idx[idx]:
            pred = row.get("eval_prediction", "")
            if pred and pred not in preds:
                preds.append(pred)
        hard_rows.append(
            {
                "sample_index": idx,
                "reviewed_primary": category,
                "target_failure_bucket": target,
                "target_rationale": rationale,
                "question": meta.get("question", ""),
                "reference_answer": meta.get("reference_answer", ""),
                "unique_predictions": " || ".join(preds[:8]),
                "image_path": meta.get("image_path", ""),
                "review_flags": meta.get("review_flags", ""),
                "issue_note": meta.get("issue_note", ""),
            }
        )

    by_bucket = Counter(row["target_failure_bucket"] for row in hard_rows)
    by_category = Counter(row["reviewed_primary"] for row in hard_rows)
    contact_sheets = build_contact_sheets(hard_rows, args.output_dir)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_or_model_used": False,
        "model_predictions_changed": False,
        "clean_after_23a_total": len(clean_indices),
        "normalized_oracle_count": len(normalized_oracle),
        "hard_failure_count": len(hard_indices),
        "hard_failure_indices": hard_indices,
        "by_reviewed_primary": dict(sorted(by_category.items())),
        "by_target_failure_bucket": dict(sorted(by_bucket.items())),
        "contact_sheets": contact_sheets,
        "outputs": {
            "hard_failure_queue_csv": str(args.output_dir / "chartqa_23b_hard_failure_queue.csv"),
            "hard_failure_queue_json": str(args.output_dir / "chartqa_23b_hard_failure_queue.json"),
            "summary_json": str(args.output_dir / "chartqa_23b_summary.json"),
            "report_md": str(args.report_md),
        },
    }

    fieldnames = [
        "sample_index",
        "reviewed_primary",
        "target_failure_bucket",
        "target_rationale",
        "question",
        "reference_answer",
        "unique_predictions",
        "image_path",
        "review_flags",
        "issue_note",
    ]
    write_csv(args.output_dir / "chartqa_23b_hard_failure_queue.csv", hard_rows, fieldnames)
    write_json(args.output_dir / "chartqa_23b_hard_failure_queue.json", hard_rows)
    write_json(args.output_dir / "chartqa_23b_summary.json", summary)
    report = build_report(summary, hard_rows)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report, encoding="utf-8")
    (args.output_dir / "chartqa_23b_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def fmt_indices(values: list[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "none"


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    category_lines = "\n".join(
        f"| `{name}` | {count} |" for name, count in summary["by_reviewed_primary"].items()
    )
    bucket_lines = "\n".join(
        f"| `{name}` | {count} |" for name, count in summary["by_target_failure_bucket"].items()
    )
    queue_lines = "\n".join(
        f"| {row['sample_index']} | `{row['target_failure_bucket']}` | `{row['reviewed_primary']}` | {row['question']} | {row['reference_answer']} |"
        for row in rows
    )
    sheets = "\n".join(f"- `{path}`" for path in summary["contact_sheets"])
    return f"""# ChartQA Module 23B hard failure targeted diagnostics - 2026-07-03

## 运行口径

Module 23B 是定向诊断准备与初筛模块：

- 不加载模型；
- 不使用 GPU；
- 不改 prediction；
- 不跑 full-val；
- 基于 Module 23A 的 clean-after-23A + normalization 后口径。

23A clean denominator 为 {summary["clean_after_23a_total"]}，normalization 后 oracle 追回 {summary["normalized_oracle_count"]}，因此本轮硬失败样本数为：

```text
{summary["hard_failure_count"]}
```

硬失败 index：

```text
{fmt_indices(summary["hard_failure_indices"])}
```

## 按人工主类别分布

| reviewed_primary | count |
|---|---:|
{category_lines}

## 按定向诊断桶分布

| target_failure_bucket | count |
|---|---:|
{bucket_lines}

## 图像复核包

已生成按 target bucket 分组的 contact sheets：

{sheets}

## Hard Failure Queue

| sample | target bucket | primary | question | reference |
|---:|---|---|---|---|
{queue_lines}

## 当前用途

这份队列用于后续人工视觉复核和 prompt/evaluator ablation 设计。23B 的下一步不是直接训练，而是逐类确认：

- 哪些是视觉读数/颜色映射失败；
- 哪些是空间语言或类别过滤失败；
- 哪些是先计算再选标签的推理失败；
- 哪些仍有残留 reference/question ambiguity。
"""


if __name__ == "__main__":
    raise SystemExit(main())
