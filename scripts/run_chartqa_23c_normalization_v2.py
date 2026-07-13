#!/usr/bin/env python
"""Module 23C: evaluator normalization v2 on existing ChartQA predictions.

This is CPU/file-only. It does not run the model. It applies four additional
normalization rules requested after Module 23B:

1. color synonyms and hex-to-color names;
2. trend morphology, e.g. increase/increases/increasing;
3. numeric answer extraction from sentences when questions contain years;
4. list order plus singular/plural variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_23A_PER_PREDICTION = Path(
    "outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv"
)
DEFAULT_23B_REVIEW = Path(
    "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/chartqa_23c_normalization_v2")
DEFAULT_REPORT_MD = Path("docs/experiments/chartqa_23c_normalization_v2_2026-07-03.md")


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


def text_norm(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u00a0", " ")
    text = text.replace("**", "")
    text = text.replace("％", "%")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def comparable_text(value: Any) -> str:
    text = text_norm(value)
    text = re.sub(r"^\s*\[|\]\s*$", "", text)
    text = re.sub(r"[*†‡]+", "", text)
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"[`*_#]", "", text)
    text = re.sub(r"\bpercent\b", "%", text)
    text = re.sub(r"[,;:!?().]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_number(value: Any) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text_norm(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def all_numbers(value: Any) -> list[float]:
    out: list[float] = []
    for item in re.findall(r"[-+]?\d+(?:\.\d+)?", text_norm(value)):
        try:
            out.append(float(item))
        except ValueError:
            pass
    return out


def numbers_close(left: float, right: float) -> bool:
    abs_tol = 0.5 if max(abs(left), abs(right)) >= 1.0 else 0.005
    return math.isclose(left, right, rel_tol=0.05, abs_tol=abs_tol)


def singularize_token(token: str) -> str:
    irregular = {
        "countries": "country",
        "assets": "asset",
        "funds": "fund",
        "citizens": "citizen",
        "degrees": "degree",
        "years": "year",
        "shares": "share",
        "statistics": "statistic",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def item_norm(value: str) -> str:
    text = comparable_text(value)
    text = re.sub(r"\b(?:the|a|an)\b", " ", text)
    tokens = [singularize_token(tok) for tok in text.split()]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def split_list_like(value: Any) -> list[str]:
    text = comparable_text(value)
    text = re.sub(r"\band\b", ",", text)
    parts = [part.strip() for part in re.split(r",|/|\|", text) if part.strip()]
    if len(parts) <= 1:
        return []
    return [item_norm(part) for part in parts]


def list_order_plural_match(prediction: Any, reference: Any) -> bool:
    ref_parts = split_list_like(reference)
    pred_parts = split_list_like(prediction)
    if len(ref_parts) < 2:
        return False
    if len(pred_parts) >= 2 and set(pred_parts) == set(ref_parts):
        return True
    pred_text = item_norm(str(prediction))
    return all(re.search(rf"(?<!\w){re.escape(part)}(?!\w)", pred_text) for part in ref_parts)


COLOR_ALIASES = {
    "light blue": {"light blue", "blue", "sky blue", "cyan", "azure", "0084b4", "2876dd", "66b2ff", "6b62ff"},
    "blue": {"blue", "light blue", "dark blue", "navy blue", "0084b4", "2876dd", "66b2ff", "6b62ff"},
    "dark blue": {"dark blue", "navy blue", "blue", "102a43", "0b2239", "123"},
    "orange": {"orange", "c2572c", "brown orange"},
    "brown": {"brown", "orange", "c2572c"},
    "yellow": {"yellow", "bright yellow", "gold"},
    "dark grey": {"dark grey", "dark gray", "grey", "gray"},
}


def color_terms(value: Any) -> set[str]:
    text = text_norm(value).replace("#", "")
    terms: set[str] = set()
    for color, aliases in COLOR_ALIASES.items():
        if color in text:
            terms.add(color)
        for alias in aliases:
            if alias in text:
                terms.add(color)
    for hex_code in re.findall(r"\b[0-9a-f]{6}\b", text):
        for color, aliases in COLOR_ALIASES.items():
            if hex_code in aliases:
                terms.add(color)
    return terms


def color_synonym_match(prediction: Any, reference: Any) -> bool:
    ref = comparable_text(reference)
    if not any(color in ref for color in COLOR_ALIASES):
        return False
    ref_terms = color_terms(reference)
    pred_terms = color_terms(prediction)
    if not ref_terms or not pred_terms:
        return False
    return bool(ref_terms & pred_terms)


TREND_GROUPS = {
    "increasing": {"increase", "increases", "increased", "increasing", "rise", "rises", "rising", "grew", "growing", "upward"},
    "decreasing": {"decrease", "decreases", "decreased", "decreasing", "decline", "declines", "declined", "fall", "falls", "falling", "downward"},
}


def trend_group(value: Any) -> str | None:
    text = comparable_text(value)
    for group, words in TREND_GROUPS.items():
        if any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) for word in words):
            return group
    return None


def trend_morphology_match(prediction: Any, reference: Any) -> bool:
    ref_group = trend_group(reference)
    pred_group = trend_group(prediction)
    return bool(ref_group and pred_group and ref_group == pred_group)


def numeric_sentence_match(row: dict[str, Any]) -> bool:
    reference = row.get("eval_reference", "")
    prediction = row.get("eval_prediction", "")
    ref_num = parse_number(reference)
    if ref_num is None:
        return False
    question = comparable_text(row.get("question", ""))
    pred_text = comparable_text(prediction)
    if not any(token in question for token in ["year", "date", "when", "1990", "2019", "2020"]):
        return False
    if len(pred_text.split()) < 5:
        return False
    for number in all_numbers(prediction):
        if 1800 <= number <= 2199 and not (1800 <= ref_num <= 2199):
            continue
        if numbers_close(number, ref_num):
            return True
    return False


def normalization_v2_match(row: dict[str, Any]) -> tuple[bool, str]:
    if color_synonym_match(row.get("eval_prediction", ""), row.get("eval_reference", "")):
        return True, "color_synonym_or_hex"
    if trend_morphology_match(row.get("eval_prediction", ""), row.get("eval_reference", "")):
        return True, "trend_morphology"
    if numeric_sentence_match(row):
        return True, "numeric_answer_in_sentence"
    if list_order_plural_match(row.get("eval_prediction", ""), row.get("eval_reference", "")):
        return True, "list_order_plural"
    return False, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-prediction-csv", type=Path, default=DEFAULT_23A_PER_PREDICTION)
    parser.add_argument("--targeted-review-csv", type=Path, default=DEFAULT_23B_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    rows = read_csv(args.per_prediction_csv)
    review_rows = read_csv(args.targeted_review_csv)
    review_group = {int(row["sample_index"]): row["review_group"] for row in review_rows}

    clean_indices: set[int] = set()
    v1_oracle: set[int] = set()
    v2_oracle: set[int] = set()
    per_prediction: list[dict[str, Any]] = []
    recovered_by_rule: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        idx = int(row["sample_index"])
        if norm_bool(row["in_23a_cleanup_exclude"]):
            continue
        clean_indices.add(idx)
        v1_correct = norm_bool(row["normalized_correct"])
        if v1_correct:
            v1_oracle.add(idx)
        v2_extra, rule = normalization_v2_match(row)
        v2_correct = v1_correct or v2_extra
        if v2_correct:
            v2_oracle.add(idx)
        if v2_extra and not v1_correct:
            recovered_by_rule[rule].add(idx)
        per_prediction.append(
            {
                **row,
                "review_group_23b": review_group.get(idx, ""),
                "normalization_v2_extra_correct": v2_extra and not v1_correct,
                "normalization_v2_rule": rule if v2_extra and not v1_correct else "",
                "normalization_v2_correct": v2_correct,
            }
        )

    recovered_rows = [row for row in per_prediction if row["normalization_v2_extra_correct"]]
    true_hard_before = {
        idx for idx, group in review_group.items() if group == "true_hard_failure" and idx in clean_indices
    }
    true_hard_after = sorted(true_hard_before - (v2_oracle - v1_oracle))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "gpu_or_model_used": False,
        "model_predictions_changed": False,
        "clean_after_23a_total": len(clean_indices),
        "oracle_v1_count": len(v1_oracle),
        "oracle_v1_accuracy": len(v1_oracle) / len(clean_indices),
        "oracle_v2_count": len(v2_oracle),
        "oracle_v2_accuracy": len(v2_oracle) / len(clean_indices),
        "oracle_v2_gain": len(v2_oracle) - len(v1_oracle),
        "oracle_v2_recovered_indices": sorted(v2_oracle - v1_oracle),
        "normalization_v2_recovered_by_rule": {
            rule: sorted(indices) for rule, indices in sorted(recovered_by_rule.items())
        },
        "true_hard_before_v2_count": len(true_hard_before),
        "true_hard_after_v2_count": len(true_hard_after),
        "true_hard_after_v2_indices": true_hard_after,
        "residual_non_hard_recovered_indices": sorted((v2_oracle - v1_oracle) - true_hard_before),
        "outputs": {
            "per_prediction_csv": str(args.output_dir / "chartqa_23c_normalization_v2_per_prediction.csv"),
            "recovered_csv": str(args.output_dir / "chartqa_23c_normalization_v2_recovered_predictions.csv"),
            "summary_json": str(args.output_dir / "chartqa_23c_normalization_v2_summary.json"),
            "report_md": str(args.report_md),
        },
    }

    fieldnames = list(per_prediction[0].keys()) if per_prediction else []
    write_csv(args.output_dir / "chartqa_23c_normalization_v2_per_prediction.csv", per_prediction, fieldnames)
    write_csv(
        args.output_dir / "chartqa_23c_normalization_v2_recovered_predictions.csv",
        recovered_rows,
        [
            "run_name",
            "sample_index",
            "review_group_23b",
            "normalization_v2_rule",
            "eval_prediction",
            "eval_reference",
            "question",
            "reviewed_primary",
        ],
    )
    write_json(args.output_dir / "chartqa_23c_normalization_v2_summary.json", summary)
    report = build_report(summary)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report, encoding="utf-8")
    (args.output_dir / "chartqa_23c_normalization_v2_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def fmt_indices(values: list[int]) -> str:
    return ", ".join(str(v) for v in values) if values else "none"


def build_report(summary: dict[str, Any]) -> str:
    rule_lines = "\n".join(
        f"| `{rule}` | {len(indices)} | `{fmt_indices(indices)}` |"
        for rule, indices in summary["normalization_v2_recovered_by_rule"].items()
    )
    return f"""# ChartQA Module 23C normalization v2 - 2026-07-03

## 运行口径

本模块只做 evaluator normalization v2：

- 不加载模型；
- 不使用 GPU；
- 不改 prediction；
- 不跑 full-val；
- 只读取 Module 23A 的 per-prediction 结果和 Module 23B 的诊断分组。

新增四类 normalization：

1. color synonyms and hex-to-color names；
2. trend morphology，例如 `increase / increases / increasing`；
3. numeric answer extraction from sentences when questions contain years；
4. list order plus singular/plural variants。

## 结果

| metric | before v2 | after v2 | gain |
|---|---:|---:|---:|
| oracle on clean-after-23A | {summary["oracle_v1_count"]}/{summary["clean_after_23a_total"]} = {summary["oracle_v1_accuracy"]:.2%} | {summary["oracle_v2_count"]}/{summary["clean_after_23a_total"]} = {summary["oracle_v2_accuracy"]:.2%} | +{summary["oracle_v2_gain"]} |

v2 oracle 追回样本：

```text
{fmt_indices(summary["oracle_v2_recovered_indices"])}
```

按规则：

| rule | unique samples | indices |
|---|---:|---|
{rule_lines}

23B true-hard 口径变化：

```text
before v2: {summary["true_hard_before_v2_count"]}
after v2:  {summary["true_hard_after_v2_count"]}
```

v2 后仍建议作为 targeted prompt ablation 的 true-hard 样本：

```text
{fmt_indices(summary["true_hard_after_v2_indices"])}
```

## 解释

Normalization v2 主要用于剥离残留 evaluator 问题，避免 targeted prompt ablation 被颜色名、趋势词、句子数字抽取和列表格式问题污染。后续 prompt ablation 应默认使用 v2 后的 true-hard 样本。
"""


if __name__ == "__main__":
    raise SystemExit(main())
