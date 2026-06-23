import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ChartQA error JSONL records by simple error type.")
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-examples-per-type", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without reading or writing error files.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def text_value(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None:
        return ""
    return str(value)


def parse_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def classify_error(record: dict[str, Any]) -> str:
    question = text_value(record, "question").lower()
    prediction = text_value(record, "eval_prediction")
    reference = text_value(record, "eval_reference")
    normalized_prediction = text_value(record, "normalized_prediction")
    normalized_reference = text_value(record, "normalized_reference")

    pred_number = parse_number(prediction)
    ref_number = parse_number(reference)

    if normalized_reference in {"yes", "no"}:
        return "yes_no_error"

    if "color" in question or "colour" in question:
        return "color_or_legend_error"

    if any(word in question for word in ["year", "date", "when"]):
        return "date_or_axis_label_error"

    if pred_number is not None and ref_number is not None:
        ratio_candidates = []
        if ref_number != 0:
            ratio_candidates.append(abs(pred_number / ref_number))
        if pred_number != 0:
            ratio_candidates.append(abs(ref_number / pred_number))
        if any(90 <= ratio <= 110 for ratio in ratio_candidates):
            return "scale_or_unit_error"
        if any(word in question for word in ["sum", "total", "average", "median", "ratio", "difference", "how many"]):
            return "calculation_error"
        return "numeric_value_error"

    if any(word in question for word in ["represent", "denote", "which", "what does"]):
        return "text_or_label_error"

    if normalized_prediction != normalized_reference:
        return "text_or_label_error"

    return "other_error"


def compact_example(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_index": record.get("sample_index"),
        "question": record.get("question"),
        "prediction": record.get("eval_prediction"),
        "reference": record.get("eval_reference"),
        "human_or_machine": record.get("human_or_machine"),
    }


def analyze(records: list[dict[str, Any]], max_examples_per_type: int) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        error_type = classify_error(record)
        counts[error_type] += 1
        bucket = examples.setdefault(error_type, [])
        if len(bucket) < max_examples_per_type:
            bucket.append(compact_example(record))

    total = len(records)
    return {
        "total_errors": total,
        "error_type_counts": dict(counts.most_common()),
        "error_type_rates": {
            error_type: count / total if total else 0.0
            for error_type, count in counts.most_common()
        },
        "examples": examples,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_path = args.output or args.errors.with_name(f"{args.errors.stem}_analysis.json")

    if args.dry_run:
        print("Dry run OK.")
        print("errors:", args.errors)
        print("output:", output_path)
        print("max_examples_per_type:", args.max_examples_per_type)
        return 0

    records = load_jsonl(args.errors)
    summary = analyze(records, args.max_examples_per_type)
    write_json(output_path, summary)

    print("Errors:", args.errors)
    print("Total errors:", summary["total_errors"])
    print("Analysis output:", output_path)
    print("\nError type counts:")
    for error_type, count in summary["error_type_counts"].items():
        print(f"- {error_type}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
