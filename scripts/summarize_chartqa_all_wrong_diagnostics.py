import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "total",
        "relaxed_correct",
        "relaxed_accuracy",
        "exact_correct",
        "exact_accuracy",
        "recovered_indices",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def find_evaluated_files(output_dir: Path) -> list[Path]:
    roots = [
        output_dir / "evaluated",
        output_dir / "table_qa_evaluated",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.glob("*_evaluated.jsonl")))
    return files


def run_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith("_evaluated.jsonl"):
        return name[: -len("_evaluated.jsonl")]
    return path.stem


def summarize_run(run_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    recovered = [row for row in rows if row.get("eval_relaxed_correct")]
    exact = [row for row in rows if row.get("eval_exact_match")]
    by_primary: dict[str, Counter] = defaultdict(Counter)
    by_flag: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        primary = row.get("reviewed_primary", "unknown")
        by_primary[primary]["total"] += 1
        by_primary[primary]["relaxed_correct"] += int(bool(row.get("eval_relaxed_correct")))
        for flag in row.get("review_flags", []) or []:
            by_flag[flag]["total"] += 1
            by_flag[flag]["relaxed_correct"] += int(bool(row.get("eval_relaxed_correct")))
    return {
        "run_name": run_name,
        "total": total,
        "relaxed_correct": len(recovered),
        "relaxed_accuracy": len(recovered) / total if total else 0.0,
        "exact_correct": len(exact),
        "exact_accuracy": len(exact) / total if total else 0.0,
        "recovered_indices": [int(row["sample_index"]) for row in recovered],
        "by_reviewed_primary": {key: dict(value) for key, value in by_primary.items()},
        "by_review_flags": {key: dict(value) for key, value in by_flag.items()},
    }


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ChartQA All-Wrong Diagnostic Subset Summary",
        "",
        f"Subset total: {summary['subset_total']}",
        "",
        "## Run Results",
        "",
        "| run | relaxed | exact |",
        "|---|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| `{run['run_name']}` | "
            f"{run['relaxed_correct']}/{run['total']} = {run['relaxed_accuracy']:.2%} | "
            f"{run['exact_correct']}/{run['total']} = {run['exact_accuracy']:.2%} |"
        )

    lines.extend(["", "## Recovered Sample Indices", ""])
    for run in summary["runs"]:
        recovered = ", ".join(str(item) for item in run["recovered_indices"]) or "none"
        lines.append(f"- `{run['run_name']}`: {recovered}")

    lines.extend(["", "## Notes", ""])
    lines.append(
        "All samples in this subset were missed by every original full-val run. "
        "A relaxed-correct sample here is therefore a recovered all-wrong case under the diagnostic condition."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ChartQA all-wrong diagnostic subset outputs.")
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chartqa_all_wrong_diagnostics"))
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_dir = args.summary_dir or args.output_dir / "summaries"
    print("subset_jsonl:", args.subset_jsonl)
    print("output_dir:", args.output_dir)
    print("summary_dir:", summary_dir)
    print("drive_output_dir:", args.drive_output_dir or "skipped")

    if args.dry_run:
        print("Dry run OK.")
        return 0

    subset_rows = read_jsonl(args.subset_jsonl)
    evaluated_files = find_evaluated_files(args.output_dir)
    if not evaluated_files:
        raise FileNotFoundError(f"No evaluated outputs found under {args.output_dir}")

    run_summaries = []
    for path in evaluated_files:
        rows = read_jsonl(path)
        run_summaries.append(summarize_run(run_name_from_path(path), rows))

    summary = {
        "subset_total": len(subset_rows),
        "evaluated_files": [str(path) for path in evaluated_files],
        "runs": run_summaries,
    }

    json_path = summary_dir / "chartqa_all_wrong_diagnostic_subset_summary.json"
    csv_path = summary_dir / "chartqa_all_wrong_diagnostic_subset_summary.csv"
    md_path = summary_dir / "chartqa_all_wrong_diagnostic_subset_report.md"
    write_json(json_path, summary)
    write_csv(csv_path, run_summaries)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(build_markdown(summary), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

    if args.drive_output_dir:
        args.drive_output_dir.mkdir(parents=True, exist_ok=True)
        for path in [json_path, csv_path, md_path]:
            shutil.copy2(path, args.drive_output_dir / path.name)
            print(f"Copied to Drive: {args.drive_output_dir / path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
