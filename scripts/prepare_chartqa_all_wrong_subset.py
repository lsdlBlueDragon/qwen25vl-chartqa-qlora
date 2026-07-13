import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def read_jsonl_by_index(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc=f"Reading {path.name}", unit="rows"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[int(row["sample_index"])] = row
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_flags(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the ChartQA all-wrong diagnostic subset JSONL.")
    parser.add_argument("--subset-csv", type=Path, required=True)
    parser.add_argument("--manual-audit-csv", type=Path, required=True)
    parser.add_argument("--full-val-sft-jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/diagnostics/chartqa_all_wrong_diagnostic_subset_85_summary.json"),
    )
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("subset_csv:", args.subset_csv)
    print("manual_audit_csv:", args.manual_audit_csv)
    print("full_val_sft_jsonl:", args.full_val_sft_jsonl)
    print("image_root:", args.image_root)
    print("output_jsonl:", args.output_jsonl)
    print("summary_output:", args.summary_output)
    print("drive_output_dir:", args.drive_output_dir or "skipped")

    if args.dry_run:
        print("Dry run OK.")
        return 0

    for path in [args.subset_csv, args.manual_audit_csv, args.full_val_sft_jsonl, args.image_root]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    subset_rows = read_csv(args.subset_csv)
    audit_rows = {int(row["sample_index"]): row for row in read_csv(args.manual_audit_csv)}
    sft_by_index = read_jsonl_by_index(args.full_val_sft_jsonl)

    records: list[dict[str, Any]] = []
    missing_images: list[str] = []
    for subset_row in tqdm(subset_rows, desc="Building diagnostic subset", unit="samples"):
        sample_index = int(subset_row["sample_index"])
        if sample_index not in audit_rows:
            raise KeyError(f"sample_index {sample_index} missing from manual audit CSV")
        if sample_index not in sft_by_index:
            raise KeyError(f"sample_index {sample_index} missing from full-val SFT JSONL")

        audit = audit_rows[sample_index]
        sft = sft_by_index[sample_index]
        image_rel = sft.get("image") or f"chartqa_val_full_sft_1920_images/val_{sample_index:06d}.png"
        image_path = args.image_root / Path(image_rel).name
        if not image_path.exists():
            missing_images.append(str(image_path))

        record = {
            "sample_index": sample_index,
            "split": sft.get("split", "val"),
            "human_or_machine": int(sft.get("human_or_machine", audit.get("human_or_machine", -1))),
            "question": sft.get("query") or audit["question"],
            "reference_answer": sft.get("answer") or audit["reference"],
            "all_labels": sft.get("all_labels") or [audit["reference"]],
            "image": image_rel,
            "image_path": str(image_path),
            "image_width": int(audit.get("image_width") or 0),
            "image_height": int(audit.get("image_height") or 0),
            "top_pred": audit.get("top_pred", ""),
            "top_count": int(audit.get("top_count") or 0),
            "uniq_preds": int(audit.get("uniq_preds") or 0),
            "original_primary": audit.get("original_primary", ""),
            "reviewed_primary": audit.get("reviewed_primary", ""),
            "primary_changed": str(audit.get("primary_changed", "")).lower() == "true",
            "review_flags": split_flags(audit.get("review_flags")),
            "issue_note": audit.get("issue_note", ""),
            "consensus": audit.get("consensus", ""),
            "recommended_action": audit.get("recommended_action", ""),
            "existing_predictions": {
                key.removeprefix("pred_"): value
                for key, value in audit.items()
                if key.startswith("pred_")
            },
        }
        records.append(record)

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"Missing {len(missing_images)} images. First missing paths:\n{preview}")

    summary = {
        "total": len(records),
        "subset_csv": str(args.subset_csv),
        "manual_audit_csv": str(args.manual_audit_csv),
        "full_val_sft_jsonl": str(args.full_val_sft_jsonl),
        "image_root": str(args.image_root),
        "reviewed_primary_counts": dict(Counter(row["reviewed_primary"] for row in records)),
        "review_flag_counts": dict(Counter(flag for row in records for flag in row["review_flags"])),
        "human_or_machine_counts": dict(Counter(str(row["human_or_machine"]) for row in records)),
    }

    write_jsonl(args.output_jsonl, records)
    write_json(args.summary_output, summary)
    print(f"Wrote {len(records)} records to {args.output_jsonl}")
    print(f"Wrote summary to {args.summary_output}")

    if args.drive_output_dir:
        args.drive_output_dir.mkdir(parents=True, exist_ok=True)
        for path in [args.output_jsonl, args.summary_output]:
            shutil.copy2(path, args.drive_output_dir / path.name)
            print(f"Copied to Drive: {args.drive_output_dir / path.name}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
