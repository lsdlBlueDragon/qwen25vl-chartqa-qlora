#!/usr/bin/env python
"""Module 23A: cleanup list + normalization-only ablation for ChartQA subset.

This module is deliberately CPU/file-only. It reads existing evaluated JSONL
files from Module 21 and 22B, applies a small hand-reviewed cleanup list, and
re-evaluates predictions with stricter answer normalization rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SUBSET_JSONL = Path("data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl")
DEFAULT_22A_EXCLUDE_CSV = Path(
    "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv"
)
DEFAULT_M21_EVAL_DIR = Path("outputs/chartqa_all_wrong_diagnostics_from_drive/evaluated")
DEFAULT_M22B_EVAL_DIR = Path("outputs/chartqa_all_wrong_diagnostics_from_drive/staged_extraction")
DEFAULT_OUTPUT_DIR = Path("outputs/chartqa_23a_cleanup_normalization")
DEFAULT_REPORT_MD = Path("docs/experiments/chartqa_23a_cleanup_normalization_2026-07-03.md")

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

# From docs/experiments/chartqa_22c_codex_visual_review_2026-07-03.md.
CODEX_CLEANUP_INDICES = {
    138: "reference_wrong_smallest_value",
    317: "reference_wrong_yes_no",
    362: "list_answer_evaluator_format",
    779: "reference_wrong_difference_question",
    882: "star_year_evaluator_format",
    946: "reference_wrong_average_comparison",
    977: "reference_question_mismatch",
    987: "question_reference_ambiguous",
    1065: "close_numeric_percent_evaluator",
    1190: "question_reference_ambiguous",
}

AMBIGUOUS_REFERENCE_SENSITIVE = {
    245: "peak_year_visually_ambiguous",
    832: "minus_sign_or_absolute_difference_ambiguous",
}


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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_22a_exclude(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(row["sample_index"]): row for row in rows}


def simple_norm(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u00a0", " ")
    text = text.replace("**", "")
    text = text.replace("％", "%")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def comparable_text(value: Any) -> str:
    text = simple_norm(value)
    text = re.sub(r"^\s*\[|\]\s*$", "", text)
    text = re.sub(r"[*†‡]+", "", text)
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"[`*_#]", "", text)
    text = re.sub(r"\bpercent\b", "%", text)
    text = re.sub(r"[,;:!?().]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_number(value: Any) -> float | None:
    text = simple_norm(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def looks_like_year(value: Any) -> bool:
    return re.fullmatch(r"(?:18|19|20|21)\d{2}\*?", simple_norm(value)) is not None


def is_year_question(question: Any, reference: Any) -> bool:
    q = comparable_text(question)
    return any(token in q for token in ["year", "date", "when"]) or looks_like_year(reference)


def strip_star_year(value: Any) -> str:
    return re.sub(r"\b((?:18|19|20|21)\d{2})\*+\b", r"\1", simple_norm(value))


def split_list_like(value: Any) -> list[str]:
    text = comparable_text(value)
    text = re.sub(r"\band\b", ",", text)
    parts = [part.strip() for part in re.split(r",|/|\|", text) if part.strip()]
    if len(parts) <= 1:
        return []
    return [re.sub(r"^(?:the|a|an)\s+", "", re.sub(r"\s+", " ", part)).strip() for part in parts]


def list_answer_match(prediction: Any, reference: Any) -> bool:
    ref_parts = split_list_like(reference)
    pred_parts = split_list_like(prediction)
    if len(ref_parts) < 2 or len(pred_parts) < 2:
        return False
    return set(ref_parts) == set(pred_parts)


def close_numeric_match(prediction: Any, reference: Any) -> bool:
    pred_num = parse_number(prediction)
    ref_num = parse_number(reference)
    if pred_num is None or ref_num is None:
        return False
    candidates = [
        (pred_num, ref_num),
        (pred_num / 100.0, ref_num),
        (pred_num, ref_num / 100.0),
        (pred_num * 100.0, ref_num),
        (pred_num, ref_num * 100.0),
    ]
    for left, right in candidates:
        # A 0.xx reference is often a percent-scale answer, but a large absolute
        # tolerance would wrongly accept 0.19 vs 0.12. Use a tight tolerance for
        # fractional comparisons and a 0.5-point tolerance for percent-sized ones.
        abs_tol = 0.5 if max(abs(left), abs(right)) >= 1.0 else 0.005
        if math.isclose(left, right, rel_tol=0.05, abs_tol=abs_tol):
            return True
    return False


def categorical_containment_match(prediction: Any, reference: Any) -> bool:
    ref = comparable_text(reference)
    pred = comparable_text(prediction)
    uncertainty_patterns = [
        "cannot determine",
        "can not determine",
        "not directly provided",
        "not enough information",
        "dont know",
        "don't know",
        "cannot be determined",
    ]
    if any(pattern in pred for pattern in uncertainty_patterns):
        return False
    if not ref or ref in {"yes", "no", "true", "false", "both", "dont know", "don't know"}:
        return False
    if parse_number(reference) is not None:
        return False
    if len(ref) < 4:
        return False
    return re.search(rf"(?<!\w){re.escape(ref)}(?!\w)", pred) is not None


def normalized_match(row: dict[str, Any]) -> tuple[bool, str, str, str]:
    prediction = row.get("eval_prediction", row.get("answer", ""))
    reference = row.get("eval_reference", row.get("reference_answer", ""))
    question = row.get("question", "")

    pred_text = comparable_text(prediction)
    ref_text = comparable_text(reference)
    if pred_text == ref_text:
        return True, "exact_after_text_cleanup", pred_text, ref_text

    pred_star = comparable_text(strip_star_year(prediction))
    ref_star = comparable_text(strip_star_year(reference))
    if pred_star == ref_star and looks_like_year(reference):
        return True, "star_year", pred_star, ref_star

    if list_answer_match(prediction, reference):
        return True, "list_format", pred_text, ref_text

    if not is_year_question(question, reference) and close_numeric_match(prediction, reference):
        return True, "close_numeric_or_percent", pred_text, ref_text

    if categorical_containment_match(prediction, reference):
        return True, "categorical_answer_contained_in_sentence", pred_text, ref_text

    return False, "", pred_text, ref_text


def load_runs(m21_dir: Path, m22b_dir: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for file_name in M21_RUN_FILES:
        path = m21_dir / file_name
        if path.exists():
            rows = read_jsonl(path)
            if rows:
                runs[rows[0].get("run_name") or file_name.replace("_evaluated.jsonl", "")] = rows
    for file_name in M22B_RUN_FILES:
        path = m22b_dir / file_name
        if path.exists():
            rows = read_jsonl(path)
            if rows:
                runs[rows[0].get("run_name") or file_name.replace("_evaluated.jsonl", "")] = rows
    return runs


def pct(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-jsonl", type=Path, default=DEFAULT_SUBSET_JSONL)
    parser.add_argument("--exclude-22a-csv", type=Path, default=DEFAULT_22A_EXCLUDE_CSV)
    parser.add_argument("--module21-eval-dir", type=Path, default=DEFAULT_M21_EVAL_DIR)
    parser.add_argument("--module22b-eval-dir", type=Path, default=DEFAULT_M22B_EVAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    subset_rows = {int(row["sample_index"]): row for row in read_jsonl(args.subset_jsonl)}
    exclude_22a = load_22a_exclude(args.exclude_22a_csv)
    valid77 = sorted(idx for idx in subset_rows if idx not in exclude_22a)
    expanded_exclude = {**{idx: "22a_exclude_or_fix_reference" for idx in exclude_22a}, **CODEX_CLEANUP_INDICES}
    clean_after_23a = sorted(idx for idx in subset_rows if idx not in expanded_exclude)

    runs = load_runs(args.module21_eval_dir, args.module22b_eval_dir)
    per_prediction: list[dict[str, Any]] = []
    run_summary: list[dict[str, Any]] = []
    oracle_base: set[int] = set()
    oracle_norm: set[int] = set()
    oracle_clean_base: set[int] = set()
    oracle_clean_norm: set[int] = set()
    recovered_by_rule: dict[str, set[int]] = defaultdict(set)

    for run_name, rows in runs.items():
        by_idx = {int(row["sample_index"]): row for row in rows}
        base_correct = 0
        norm_correct = 0
        clean_base_correct = 0
        clean_norm_correct = 0
        norm_recovered_indices: list[int] = []

        for idx in valid77:
            row = by_idx.get(idx)
            if not row:
                continue
            base = bool(row.get("eval_relaxed_correct"))
            norm, rule, norm_pred, norm_ref = normalized_match(row)
            final = base or norm
            if base:
                base_correct += 1
                oracle_base.add(idx)
            if final:
                norm_correct += 1
                oracle_norm.add(idx)
            if idx in clean_after_23a:
                if base:
                    clean_base_correct += 1
                    oracle_clean_base.add(idx)
                if final:
                    clean_norm_correct += 1
                    oracle_clean_norm.add(idx)
            if final and not base:
                norm_recovered_indices.append(idx)
                recovered_by_rule[rule].add(idx)

            per_prediction.append(
                {
                    "run_name": run_name,
                    "sample_index": idx,
                    "in_23a_cleanup_exclude": idx in CODEX_CLEANUP_INDICES,
                    "cleanup_reason": CODEX_CLEANUP_INDICES.get(idx, ""),
                    "ambiguous_reference_sensitive": idx in AMBIGUOUS_REFERENCE_SENSITIVE,
                    "ambiguous_reason": AMBIGUOUS_REFERENCE_SENSITIVE.get(idx, ""),
                    "base_relaxed_correct": base,
                    "normalized_correct": final,
                    "normalization_recovered": final and not base,
                    "normalization_rule": rule if final and not base else "",
                    "eval_prediction": row.get("eval_prediction", ""),
                    "eval_reference": row.get("eval_reference", ""),
                    "normalized_prediction_23a": norm_pred,
                    "normalized_reference_23a": norm_ref,
                    "question": row.get("question", ""),
                    "reviewed_primary": row.get("reviewed_primary", subset_rows[idx].get("reviewed_primary", "")),
                }
            )

        clean_total = len(clean_after_23a)
        run_summary.append(
            {
                "run_name": run_name,
                "valid77_base_correct": base_correct,
                "valid77_base_accuracy": pct(base_correct, len(valid77)),
                "valid77_normalized_correct": norm_correct,
                "valid77_normalized_accuracy": pct(norm_correct, len(valid77)),
                "valid77_normalization_gain": norm_correct - base_correct,
                "valid77_normalization_recovered_indices": ",".join(map(str, sorted(norm_recovered_indices))),
                "clean_after_23a_total": clean_total,
                "clean_after_23a_base_correct": clean_base_correct,
                "clean_after_23a_base_accuracy": pct(clean_base_correct, clean_total),
                "clean_after_23a_normalized_correct": clean_norm_correct,
                "clean_after_23a_normalized_accuracy": pct(clean_norm_correct, clean_total),
                "clean_after_23a_normalization_gain": clean_norm_correct - clean_base_correct,
            }
        )

    cleanup_rows = []
    for idx, reason in sorted(expanded_exclude.items()):
        source = "22C_codex_visual_review" if idx in CODEX_CLEANUP_INDICES else "22A_evaluator_cleanup"
        cleanup_rows.append(
            {
                "sample_index": idx,
                "source": source,
                "reason": reason,
                "question": subset_rows.get(idx, {}).get("question", ""),
                "reference_answer": subset_rows.get(idx, {}).get("reference_answer", ""),
                "reviewed_primary": subset_rows.get(idx, {}).get("reviewed_primary", ""),
            }
        )

    norm_recovered_rows = [row for row in per_prediction if row["normalization_recovered"]]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_or_model_used": False,
        "model_predictions_changed": False,
        "subset85_total": len(subset_rows),
        "exclude_22a_count": len(exclude_22a),
        "exclude_22a_indices": sorted(exclude_22a),
        "valid77_total": len(valid77),
        "codex_cleanup_count": len(CODEX_CLEANUP_INDICES),
        "codex_cleanup_indices": sorted(CODEX_CLEANUP_INDICES),
        "ambiguous_reference_sensitive_indices": sorted(AMBIGUOUS_REFERENCE_SENSITIVE),
        "expanded_exclude_count": len(expanded_exclude),
        "expanded_exclude_indices": sorted(expanded_exclude),
        "clean_after_23a_total": len(clean_after_23a),
        "valid77_oracle_base_count": len(oracle_base),
        "valid77_oracle_base_accuracy": pct(len(oracle_base), len(valid77)),
        "valid77_oracle_normalized_count": len(oracle_norm),
        "valid77_oracle_normalized_accuracy": pct(len(oracle_norm), len(valid77)),
        "valid77_oracle_normalization_gain": len(oracle_norm) - len(oracle_base),
        "valid77_oracle_normalization_recovered_indices": sorted(oracle_norm - oracle_base),
        "clean_after_23a_oracle_base_count": len(oracle_clean_base),
        "clean_after_23a_oracle_base_accuracy": pct(len(oracle_clean_base), len(clean_after_23a)),
        "clean_after_23a_oracle_normalized_count": len(oracle_clean_norm),
        "clean_after_23a_oracle_normalized_accuracy": pct(len(oracle_clean_norm), len(clean_after_23a)),
        "clean_after_23a_oracle_normalization_gain": len(oracle_clean_norm) - len(oracle_clean_base),
        "normalization_recovered_unique_indices": sorted(
            {int(row["sample_index"]) for row in norm_recovered_rows}
        ),
        "normalization_recovered_by_rule": {
            rule: sorted(indices) for rule, indices in sorted(recovered_by_rule.items()) if rule
        },
        "run_count": len(runs),
        "outputs": {
            "cleanup_list_csv": str(args.output_dir / "chartqa_23a_expanded_cleanup_exclude_list.csv"),
            "per_prediction_csv": str(args.output_dir / "chartqa_23a_normalization_per_prediction.csv"),
            "normalization_recovered_csv": str(args.output_dir / "chartqa_23a_normalization_recovered_predictions.csv"),
            "run_summary_csv": str(args.output_dir / "chartqa_23a_run_summary.csv"),
            "summary_json": str(args.output_dir / "chartqa_23a_summary.json"),
            "report_md": str(args.report_md),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "chartqa_23a_expanded_cleanup_exclude_list.csv",
        cleanup_rows,
        ["sample_index", "source", "reason", "question", "reference_answer", "reviewed_primary"],
    )
    write_csv(
        args.output_dir / "chartqa_23a_normalization_per_prediction.csv",
        per_prediction,
        [
            "run_name",
            "sample_index",
            "in_23a_cleanup_exclude",
            "cleanup_reason",
            "ambiguous_reference_sensitive",
            "ambiguous_reason",
            "base_relaxed_correct",
            "normalized_correct",
            "normalization_recovered",
            "normalization_rule",
            "eval_prediction",
            "eval_reference",
            "normalized_prediction_23a",
            "normalized_reference_23a",
            "question",
            "reviewed_primary",
        ],
    )
    write_csv(
        args.output_dir / "chartqa_23a_normalization_recovered_predictions.csv",
        norm_recovered_rows,
        [
            "run_name",
            "sample_index",
            "in_23a_cleanup_exclude",
            "cleanup_reason",
            "normalization_rule",
            "eval_prediction",
            "eval_reference",
            "question",
            "reviewed_primary",
        ],
    )
    write_csv(
        args.output_dir / "chartqa_23a_run_summary.csv",
        run_summary,
        [
            "run_name",
            "valid77_base_correct",
            "valid77_base_accuracy",
            "valid77_normalized_correct",
            "valid77_normalized_accuracy",
            "valid77_normalization_gain",
            "valid77_normalization_recovered_indices",
            "clean_after_23a_total",
            "clean_after_23a_base_correct",
            "clean_after_23a_base_accuracy",
            "clean_after_23a_normalized_correct",
            "clean_after_23a_normalized_accuracy",
            "clean_after_23a_normalization_gain",
        ],
    )
    write_json(args.output_dir / "chartqa_23a_summary.json", summary)
    report = build_report(summary, run_summary)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report, encoding="utf-8")
    (args.output_dir / "chartqa_23a_report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def fmt_indices(values: list[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "none"


def build_report(summary: dict[str, Any], run_summary: list[dict[str, Any]]) -> str:
    run_lines = "\n".join(
        "| `{run_name}` | {valid77_base_correct}/77 | {valid77_normalized_correct}/77 | +{valid77_normalization_gain} | {clean_after_23a_base_correct}/{clean_after_23a_total} | {clean_after_23a_normalized_correct}/{clean_after_23a_total} |".format(
            **row
        )
        for row in run_summary
    )
    rule_lines = "\n".join(
        f"| `{rule}` | {len(indices)} | `{fmt_indices(indices)}` |"
        for rule, indices in summary["normalization_recovered_by_rule"].items()
    )

    return f"""# ChartQA Module 23A cleanup + normalization-only ablation - 2026-07-03

## 运行口径

Module 23A 已完成。它是纯本地后处理模块：

- 不加载模型；
- 不使用 GPU；
- 不改任何 prediction；
- 不跑 full-val；
- 只读取现有 Module 21 / 22B evaluated JSONL。

本模块分两步：

1. 应用 cleanup list：在 22A 原有 8 条 exclude 的基础上，加入 Codex 视觉复核标记的 10 条 reference/evaluator 问题样本。
2. 做 normalization-only 消融：在 22A 后的 77 条 subset 上，只改 answer normalization，观察能追回多少。

## Cleanup List

22A 原始排除：

```text
{fmt_indices(summary["exclude_22a_indices"])}
```

23A 新增 Codex cleanup：

```text
{fmt_indices(summary["codex_cleanup_indices"])}
```

23A 扩展后 exclude 总数：

```text
{summary["expanded_exclude_count"]}
```

因此 clean-after-23A denominator 为：

```text
{summary["clean_after_23a_total"]}
```

另有两个样本只标记为 ambiguous/reference-sensitive，默认不加入 exclude：

```text
{fmt_indices(summary["ambiguous_reference_sensitive_indices"])}
```

## Normalization Rules

23A 新增 normalization 只覆盖以下规则：

- list answer format：例如 `Czech Republic, New Zealand` vs `[Czech Republic, New Zealand]`；
- star year：例如 `2028* -> 2028`；
- categorical answer contained in sentence：例如句子中明确包含 `orange`；
- percent / close numeric：例如 `65%` 或 `65` vs `65.3`。

没有加入会扩大语义边界的规则，例如把 reference 错误的样本强行判对。

## 77 条上的 normalization-only 消融

| metric | before | after | gain |
|---|---:|---:|---:|
| oracle on valid77 | {summary["valid77_oracle_base_count"]}/77 = {summary["valid77_oracle_base_accuracy"]:.2%} | {summary["valid77_oracle_normalized_count"]}/77 = {summary["valid77_oracle_normalized_accuracy"]:.2%} | +{summary["valid77_oracle_normalization_gain"]} |

normalization-only 追回的 unique sample：

```text
{fmt_indices(summary["valid77_oracle_normalization_recovered_indices"])}
```

按规则分布：

| rule | unique samples | sample indices |
|---|---:|---|
{rule_lines}

## Clean Denominator 口径

应用 23A expanded cleanup 后：

| metric | before normalization | after normalization |
|---|---:|---:|
| oracle on clean-after-23A | {summary["clean_after_23a_oracle_base_count"]}/{summary["clean_after_23a_total"]} = {summary["clean_after_23a_oracle_base_accuracy"]:.2%} | {summary["clean_after_23a_oracle_normalized_count"]}/{summary["clean_after_23a_total"]} = {summary["clean_after_23a_oracle_normalized_accuracy"]:.2%} |

## Per-run Summary

| run | valid77 before | valid77 after | gain | clean before | clean after |
|---|---:|---:|---:|---:|---:|
{run_lines}

## 当前判断

23A 把两件事分开了：

- normalization-only 的收益说明有多少“答案已经基本对，但 evaluator 没吃到”；
- expanded cleanup 的收益说明当前 subset 里有多少不适合作为模型失败统计的 reference/evaluator 问题。

如果后续要做硬失败定向诊断，建议使用 clean-after-23A denominator，同时保留 ambiguous 样本单独统计。这样下一轮 strict threshold、date-axis、range aggregation、spatial grounding 等诊断不会被 reference cleanup 和 answer formatting 干扰。

## 输出文件

- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_expanded_cleanup_exclude_list.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_recovered_predictions.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_run_summary.csv`
- `outputs/chartqa_23a_cleanup_normalization/chartqa_23a_summary.json`
- `docs/experiments/chartqa_23a_cleanup_normalization_2026-07-03.md`
"""


if __name__ == "__main__":
    raise SystemExit(main())
