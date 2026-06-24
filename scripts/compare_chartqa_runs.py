import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two evaluated ChartQA JSONL files.")
    parser.add_argument("--baseline-evaluated", type=Path, required=True)
    parser.add_argument("--adapter-evaluated", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-examples-per-bucket", type=int, default=5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without reading or writing files.",
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


def record_key(record: dict[str, Any], fallback_index: int) -> str:
    if record.get("sample_index") is not None:
        return str(record["sample_index"])
    return str(fallback_index)


def compact_pair(
    baseline: dict[str, Any],
    adapter: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_index": adapter.get("sample_index", baseline.get("sample_index")),
        "question": adapter.get("question", baseline.get("question")),
        "reference": adapter.get("eval_reference", baseline.get("eval_reference")),
        "baseline_prediction": baseline.get("eval_prediction"),
        "adapter_prediction": adapter.get("eval_prediction"),
        "baseline_exact": baseline.get("eval_exact_match"),
        "adapter_exact": adapter.get("eval_exact_match"),
        "baseline_relaxed": baseline.get("eval_relaxed_correct"),
        "adapter_relaxed": adapter.get("eval_relaxed_correct"),
        "human_or_machine": adapter.get("human_or_machine", baseline.get("human_or_machine")),
    }


def compare(
    baseline_records: list[dict[str, Any]],
    adapter_records: list[dict[str, Any]],
    max_examples_per_bucket: int,
) -> dict[str, Any]:
    baseline_by_key = {
        record_key(record, index): record
        for index, record in enumerate(baseline_records)
    }

    buckets = {
        "improved": [],
        "regressed": [],
        "both_correct": [],
        "both_wrong": [],
    }
    missing_baseline = []

    for index, adapter in enumerate(adapter_records):
        key = record_key(adapter, index)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            missing_baseline.append(key)
            continue

        baseline_correct = bool(baseline.get("eval_relaxed_correct"))
        adapter_correct = bool(adapter.get("eval_relaxed_correct"))
        pair = compact_pair(baseline, adapter)

        if not baseline_correct and adapter_correct:
            buckets["improved"].append(pair)
        elif baseline_correct and not adapter_correct:
            buckets["regressed"].append(pair)
        elif baseline_correct and adapter_correct:
            buckets["both_correct"].append(pair)
        else:
            buckets["both_wrong"].append(pair)

    compared = sum(len(items) for items in buckets.values())
    counts = {name: len(items) for name, items in buckets.items()}

    return {
        "compared_records": compared,
        "missing_baseline_keys": missing_baseline,
        "bucket_counts": counts,
        "bucket_rates": {
            name: count / compared if compared else 0.0
            for name, count in counts.items()
        },
        "net_relaxed_gain": counts["improved"] - counts["regressed"],
        "examples": {
            name: items[:max_examples_per_bucket]
            for name, items in buckets.items()
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_path = args.output or args.adapter_evaluated.with_name(
        f"{args.adapter_evaluated.stem}_vs_baseline.json"
    )

    if args.dry_run:
        print("Dry run OK.")
        print("baseline_evaluated:", args.baseline_evaluated)
        print("adapter_evaluated:", args.adapter_evaluated)
        print("output:", output_path)
        print("max_examples_per_bucket:", args.max_examples_per_bucket)
        return 0

    baseline_records = load_jsonl(args.baseline_evaluated)
    adapter_records = load_jsonl(args.adapter_evaluated)
    summary = compare(baseline_records, adapter_records, args.max_examples_per_bucket)
    write_json(output_path, summary)

    print("Baseline evaluated:", args.baseline_evaluated)
    print("Adapter evaluated:", args.adapter_evaluated)
    print("Compared records:", summary["compared_records"])
    print("Comparison output:", output_path)
    print("\nBucket counts:")
    for bucket, count in summary["bucket_counts"].items():
        print(f"- {bucket}: {count}")
    print("Net relaxed gain:", summary["net_relaxed_gain"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
