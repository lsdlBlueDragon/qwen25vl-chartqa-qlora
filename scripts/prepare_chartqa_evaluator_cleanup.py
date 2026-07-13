import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_flags(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def infer_issue_type(row: dict[str, str]) -> str:
    flags = set(split_flags(row.get("review_flags")))
    note = (row.get("issue_note") or "").lower()
    reviewed = row.get("reviewed_primary", "")

    if "date_serial_reference" in flags or reviewed == "date_serial_or_label_format":
        return "date_serial_reference"
    if "list_answer_format" in flags:
        return "list_answer_format"
    if "color granularity" in note or "light blue" in note:
        return "color_granularity"
    if "scale normalization" in note or "percent-like" in note:
        return "scale_normalization"
    if "semantically equivalent" in note:
        return "semantic_equivalence"
    if "near-miss" in note or "ocr spelling" in note:
        return "ocr_spelling_near_miss"
    if "answer-type mismatch" in note or "reference error" in note or "annotation" in note:
        return "answer_type_or_reference_mismatch"
    if reviewed == "data_or_evaluator_issue" or "data_or_evaluator_issue" in flags:
        return "general_data_or_evaluator_issue"
    return "not_cleanup_candidate"


def policy_for_issue(issue_type: str) -> tuple[str, str]:
    if issue_type in {"date_serial_reference", "answer_type_or_reference_mismatch", "general_data_or_evaluator_issue"}:
        return "exclude_or_fix_reference", "high"
    if issue_type in {"semantic_equivalence", "ocr_spelling_near_miss", "color_granularity", "scale_normalization"}:
        return "normalization_candidate", "medium"
    if issue_type == "list_answer_format":
        return "answer_format_manual_review", "medium"
    return "keep_as_model_error", "low"


def is_cleanup_candidate(row: dict[str, str]) -> bool:
    flags = set(split_flags(row.get("review_flags")))
    issue_type = infer_issue_type(row)
    if issue_type != "not_cleanup_candidate":
        return True
    if row.get("issue_note", "").strip():
        return True
    if "data_or_evaluator_issue" in flags:
        return True
    return False


def build_cleanup_rows(rows: list[dict[str, str]], subset_indices: set[int] | None) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    iterable = rows
    if subset_indices is not None:
        iterable = [row for row in rows if int(row["sample_index"]) in subset_indices]

    for row in tqdm(iterable, desc="Classifying cleanup candidates", unit="samples"):
        issue_type = infer_issue_type(row)
        policy, priority = policy_for_issue(issue_type)
        flags = split_flags(row.get("review_flags"))
        cleanup_candidate = is_cleanup_candidate(row)
        output_rows.append(
            {
                "sample_index": int(row["sample_index"]),
                "in_recommended_subset": subset_indices is not None and int(row["sample_index"]) in subset_indices,
                "cleanup_candidate": cleanup_candidate,
                "issue_type": issue_type,
                "cleanup_policy": policy,
                "cleanup_priority": priority,
                "reviewed_primary": row.get("reviewed_primary", ""),
                "review_flags": ";".join(flags),
                "question": row.get("question", ""),
                "reference": row.get("reference", ""),
                "top_pred": row.get("top_pred", ""),
                "top_count": row.get("top_count", ""),
                "uniq_preds": row.get("uniq_preds", ""),
                "issue_note": row.get("issue_note", ""),
                "recommended_action": row.get("recommended_action", ""),
            }
        )
    return output_rows


def summarize(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["cleanup_candidate"]]
    return {
        "name": name,
        "total": len(rows),
        "cleanup_candidate_count": len(candidates),
        "effective_model_error_count_if_excluded": len(rows) - len(
            [row for row in candidates if row["cleanup_policy"] == "exclude_or_fix_reference"]
        ),
        "issue_type_counts": dict(Counter(row["issue_type"] for row in candidates)),
        "cleanup_policy_counts": dict(Counter(row["cleanup_policy"] for row in candidates)),
        "cleanup_priority_counts": dict(Counter(row["cleanup_priority"] for row in candidates)),
        "reviewed_primary_counts": dict(Counter(row["reviewed_primary"] for row in candidates)),
    }


def build_markdown(all_summary: dict[str, Any], subset_summary: dict[str, Any], subset_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ChartQA evaluator/data cleanup candidates - Module 22A",
        "",
        "## 目的",
        "",
        "本模块只整理评测口径、标注、reference、答案格式和 normalization 问题，不重新跑模型，不修改历史 full-val 指标。",
        "",
        "## 汇总",
        "",
        f"- all-wrong 全量审计表样本数：{all_summary['total']}",
        f"- 全量 cleanup candidates：{all_summary['cleanup_candidate_count']}",
        f"- 推荐 85 条 subset 样本数：{subset_summary['total']}",
        f"- subset cleanup candidates：{subset_summary['cleanup_candidate_count']}",
        f"- 若仅剔除 `exclude_or_fix_reference` 类，subset 有效模型错误数：{subset_summary['effective_model_error_count_if_excluded']}",
        "",
        "## subset cleanup policy 分布",
        "",
        "| policy | count |",
        "|---|---:|",
    ]
    for key, value in sorted(subset_summary["cleanup_policy_counts"].items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## subset issue type 分布",
            "",
            "| issue_type | count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(subset_summary["issue_type_counts"].items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## subset cleanup candidates",
            "",
            "| sample | issue_type | policy | reference | top_pred | note |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for row in subset_rows:
        if not row["cleanup_candidate"]:
            continue
        note = str(row["issue_note"]).replace("|", "/")
        question_note = note if note else str(row["question"]).replace("|", "/")[:120]
        lines.append(
            f"| {row['sample_index']} | `{row['issue_type']}` | `{row['cleanup_policy']}` | "
            f"`{str(row['reference']).replace('|', '/')}` | `{str(row['top_pred']).replace('|', '/')}` | {question_note} |"
        )

    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "1. `exclude_or_fix_reference`：先不要用于模型能力增益判断，进入 reference 修订或单独剔除统计。",
            "2. `normalization_candidate`：不要直接视为模型完全错误，先设计 normalization/evaluator rule，再人工抽查。",
            "3. `answer_format_manual_review`：确认题目到底要求 list、sum、任意顺序 list，避免把格式歧义当成模型失败。",
            "4. 本模块产物只定义清理清单；是否正式改 evaluator，放到后续模块或 commit 决策。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ChartQA evaluator/data cleanup candidates for Module 22A.")
    parser.add_argument("--manual-audit-csv", type=Path, required=True)
    parser.add_argument("--subset-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chartqa_evaluator_cleanup"))
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("manual_audit_csv:", args.manual_audit_csv)
    print("subset_csv:", args.subset_csv or "skipped")
    print("output_dir:", args.output_dir)
    print("drive_output_dir:", args.drive_output_dir or "skipped")

    if args.dry_run:
        print("Dry run OK.")
        return 0
    if not args.manual_audit_csv.exists():
        raise FileNotFoundError(args.manual_audit_csv)

    all_rows = read_csv(args.manual_audit_csv)
    subset_indices = None
    if args.subset_csv:
        if not args.subset_csv.exists():
            raise FileNotFoundError(args.subset_csv)
        subset_indices = {int(row["sample_index"]) for row in read_csv(args.subset_csv)}

    all_cleanup_rows = build_cleanup_rows(all_rows, subset_indices=None)
    subset_cleanup_rows = build_cleanup_rows(all_rows, subset_indices=subset_indices) if subset_indices else []

    all_summary = summarize(all_cleanup_rows, "all_wrong_325")
    subset_summary = summarize(subset_cleanup_rows, "recommended_subset_85") if subset_indices else {}
    summary = {
        "all_wrong": all_summary,
        "recommended_subset": subset_summary,
    }

    fields = [
        "sample_index",
        "in_recommended_subset",
        "cleanup_candidate",
        "issue_type",
        "cleanup_policy",
        "cleanup_priority",
        "reviewed_primary",
        "review_flags",
        "question",
        "reference",
        "top_pred",
        "top_count",
        "uniq_preds",
        "issue_note",
        "recommended_action",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_json = args.output_dir / "chartqa_all_wrong_evaluator_cleanup_candidates.json"
    all_csv = args.output_dir / "chartqa_all_wrong_evaluator_cleanup_candidates.csv"
    subset_json = args.output_dir / "chartqa_subset85_evaluator_cleanup_candidates.json"
    subset_csv = args.output_dir / "chartqa_subset85_evaluator_cleanup_candidates.csv"
    ignore_csv = args.output_dir / "chartqa_subset85_exclude_or_fix_reference_list.csv"
    normalization_csv = args.output_dir / "chartqa_subset85_normalization_candidates.csv"
    summary_json = args.output_dir / "chartqa_evaluator_cleanup_summary.json"
    report_md = args.output_dir / "chartqa_evaluator_cleanup_report.md"

    write_json(all_json, all_cleanup_rows)
    write_csv(all_csv, all_cleanup_rows, fields)
    if subset_indices:
        write_json(subset_json, subset_cleanup_rows)
        write_csv(subset_csv, subset_cleanup_rows, fields)
        write_csv(
            ignore_csv,
            [row for row in subset_cleanup_rows if row["cleanup_policy"] == "exclude_or_fix_reference"],
            fields,
        )
        write_csv(
            normalization_csv,
            [row for row in subset_cleanup_rows if row["cleanup_policy"] in {"normalization_candidate", "answer_format_manual_review"}],
            fields,
        )
        report_md.write_text(build_markdown(all_summary, subset_summary, subset_cleanup_rows), encoding="utf-8")
    write_json(summary_json, summary)

    outputs = [all_json, all_csv, summary_json]
    if subset_indices:
        outputs.extend([subset_json, subset_csv, ignore_csv, normalization_csv, report_md])

    for path in outputs:
        print(f"Wrote {path}")

    if args.drive_output_dir:
        args.drive_output_dir.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            shutil.copy2(path, args.drive_output_dir / path.name)
            print(f"Copied to Drive: {args.drive_output_dir / path.name}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
