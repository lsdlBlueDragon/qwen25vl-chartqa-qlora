"""Audit ChartQA full-val samples missed by every available 3B run.

The script reads the final full-val evaluated JSONL files, reconstructs the
all-runs-wrong set, and writes compact audit artifacts for manual/model review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path


RUN_FILES = {
    "baseline_default": "chartqa_val_full_3b_baseline_default_1920_evaluated.jsonl",
    "standard_steps100": "chartqa_val_full_3b_standard_steps100_1920_evaluated.jsonl",
    "standard_numeric_final": "chartqa_val_full_3b_standard_steps100_numeric_final_1920_evaluated.jsonl",
    "experiment_a_steps200": "chartqa_val_full_3b_steps200_1920_evaluated.jsonl",
    "experiment_b_calcnum": "chartqa_val_full_3b_calcnum1k_steps100_1920_evaluated.jsonl",
    "experiment_d_hardmix": "chartqa_val_full_3b_hardmix1k_steps100_1920_evaluated.jsonl",
    "experiment_f_steps250_r16a32": "chartqa_val_full_3b_steps250_r16a32_1920_evaluated.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def is_number(value: object) -> bool:
    text = str(value or "").strip().replace(",", "")
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?%?", text))


def is_excel_date_serial(value: object) -> bool:
    if not is_number(value):
        return False
    try:
        number = float(str(value).replace(",", "").replace("%", ""))
    except ValueError:
        return False
    return 30000 <= number <= 50000 and number.is_integer()


def classify(row: dict, predictions: dict[str, str]) -> tuple[str, list[str], str]:
    question = norm_text(row.get("question"))
    reference = row.get("reference_answer", row.get("answer"))
    reference_text = norm_text(reference)
    pred_values = [norm_text(value) for value in predictions.values()]
    pred_counts = Counter(pred_values)
    top_prediction, top_count = pred_counts.most_common(1)[0]

    tags: list[str] = []

    date_terms = [
        "month",
        "date",
        "when",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sept",
        "sep",
        "oct",
        "nov",
        "dec",
        "quarter",
    ]
    calc_terms = [
        "difference",
        "sum",
        "total",
        "average",
        "increased",
        "decreased",
        "increase",
        "decrease",
        "more than",
        "less than",
        "greater than",
        "minus",
        "plus",
        "combined",
        "altogether",
        "ratio",
        "percent",
        "percentage",
        "times",
    ]
    count_terms = ["how many", "number of", "count"]
    visual_terms = [
        "color",
        "blue",
        "green",
        "red",
        "yellow",
        "legend",
        "line",
        "bar",
        "graph",
        "series",
        "curve",
    ]
    extreme_terms = [
        "highest",
        "lowest",
        "largest",
        "smallest",
        "maximum",
        "minimum",
        "peak",
        "top",
        "least",
        "most",
    ]
    value_terms = [
        "value",
        "how much",
        "what is the",
        "what was the",
        "amount",
        "sales",
        "rate",
        "price",
        "population",
    ]

    has_date_axis = (
        any(term in question for term in date_terms)
        or bool(re.search(r"\b(in|from|to|since|until|by)\s+\d{4}\b", question))
        or bool(re.search(r"\b(which|what|in what)\s+year\b", question))
        or bool(re.search(r"\byear\s+(did|does|was|were|recorded|had|has|shows?)\b", question))
    )
    expects_date_output = (
        question.startswith("when ")
        or bool(re.search(r"\b(which|what|in what)\s+year\b", question))
        or "which month" in question
        or "what month" in question
    )
    has_calculation = any(term in question for term in calc_terms)

    if is_excel_date_serial(reference) and has_date_axis:
        tags.append("date_serial_or_label_format")
    if has_date_axis:
        tags.append("date_axis_reading")
    if has_calculation:
        tags.append("multi_step_calculation")
    if any(term in question for term in count_terms):
        tags.append("counting_or_category_count")
    if question.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ")):
        tags.append("yes_no_or_boolean")
    if any(term in question for term in visual_terms):
        tags.append("visual_mapping_or_legend")
    if any(term in question for term in extreme_terms):
        tags.append("extreme_value_or_ranking")
    if any(term in question for term in value_terms) or is_number(reference):
        tags.append("numeric_value_or_scale")
    if not tags:
        tags.append("text_label_lookup")

    if top_count >= 5:
        tags.append("shared_wrong_consensus")
    elif len(pred_counts) >= 5:
        tags.append("unstable_across_runs")

    if reference_text in pred_counts:
        tags.append("normalization_or_evaluator_edge_case")

    if "date_serial_or_label_format" in tags:
        primary = "date_serial_or_label_format"
    elif has_calculation and not expects_date_output:
        primary = "multi_step_calculation"
    elif expects_date_output and "date_axis_reading" in tags:
        primary = "date_axis_reading"
    else:
        primary_priority = [
            "date_axis_reading",
            "counting_or_category_count",
            "yes_no_or_boolean",
            "visual_mapping_or_legend",
            "extreme_value_or_ranking",
            "numeric_value_or_scale",
            "text_label_lookup",
        ]
        primary = next((tag for tag in primary_priority if tag in tags), tags[0])

    if top_count >= 5:
        consensus = f"high consensus wrong: {top_count}/{len(predictions)} predicted {top_prediction!r}"
    elif len(pred_counts) >= 5:
        consensus = f"high disagreement: {len(pred_counts)} unique predictions"
    else:
        consensus = f"mixed: {len(pred_counts)} unique predictions, top {top_count}/{len(predictions)}"

    return primary, tags, consensus


def write_contact_sheets(records: list[dict], image_root: Path, out_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - optional diagnostic output
        print(f"Skipping contact sheets because Pillow is unavailable: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    thumb_w, thumb_h = 360, 240
    text_h = 170
    cols = 2
    rows_per_sheet = 5
    cell_w, cell_h = thumb_w, thumb_h + text_h

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["primary_error_type"]].append(record)

    for category, items in sorted(grouped.items()):
        sample_items = items[:10]
        sheet_rows = (len(sample_items) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, sheet_rows * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, item in enumerate(sample_items):
            x = (idx % cols) * cell_w
            y = (idx // cols) * cell_h
            image_path = image_root / Path(item["image_path"]).name
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb_w, thumb_h))
                sheet.paste(image, (x, y))
            except Exception:
                draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline="red")
                draw.text((x + 8, y + 8), f"Missing image: {image_path.name}", fill="red", font=font)

            text = (
                f"idx={item['sample_index']} ref={item['reference']}\n"
                f"Q: {item['question']}\n"
                f"Top: {item['consensus_note']}"
            )
            wrapped = []
            for line in text.splitlines():
                wrapped.extend(textwrap.wrap(line, width=58) or [""])
            draw.multiline_text((x + 4, y + thumb_h + 4), "\n".join(wrapped[:9]), fill="black", font=font)

        sheet.save(out_dir / f"{category}_sample_contact_sheet.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive-root",
        default=r"G:\我的云端硬盘\qwen25vl-chartqa-qlora",
        help="Local Google Drive project root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/chartqa_all_runs_wrong_audit",
        help="Workspace output directory for audit artifacts.",
    )
    parser.add_argument("--contact-sheets", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_root)
    full_val_dir = drive_root / "outputs" / "chartqa_3b_full_val"
    image_root = drive_root / "data" / "processed" / "chartqa_val_full_sft_1920_images"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_run: dict[str, dict[int, dict]] = {}
    for run_name, file_name in RUN_FILES.items():
        rows = read_jsonl(full_val_dir / file_name)
        by_run[run_name] = {int(row["sample_index"]): row for row in rows}

    common_indices = set.intersection(*(set(rows) for rows in by_run.values()))
    all_wrong_indices = []
    for sample_index in sorted(common_indices):
        if all(not by_run[run][sample_index].get("eval_relaxed_correct", False) for run in RUN_FILES):
            all_wrong_indices.append(sample_index)

    records = []
    for sample_index in all_wrong_indices:
        base = by_run["baseline_default"][sample_index]
        predictions = {
            run: by_run[run][sample_index].get("eval_prediction", "")
            for run in RUN_FILES
        }
        primary, tags, consensus = classify(base, predictions)
        record = {
            "sample_index": sample_index,
            "human_or_machine": base.get("human_or_machine"),
            "split": base.get("split"),
            "question": base.get("question", ""),
            "reference": base.get("reference_answer", base.get("answer", "")),
            "image_path": base.get("image_path", ""),
            "absolute_image_path": str(image_root / Path(base.get("image_path", "")).name),
            "primary_error_type": primary,
            "audit_tags": tags,
            "consensus_note": consensus,
            "predictions": predictions,
        }
        for run, prediction in predictions.items():
            record[f"pred_{run}"] = prediction
        records.append(record)

    summary = {
        "total_all_runs_wrong": len(records),
        "runs": list(RUN_FILES),
        "primary_error_type_counts": Counter(row["primary_error_type"] for row in records),
        "tag_counts": Counter(tag for row in records for tag in row["audit_tags"]),
        "human_or_machine_counts": Counter(str(row["human_or_machine"]) for row in records),
        "high_consensus_wrong_count": sum("shared_wrong_consensus" in row["audit_tags"] for row in records),
        "unstable_across_runs_count": sum("unstable_across_runs" in row["audit_tags"] for row in records),
    }

    json_path = out_dir / "chartqa_full_val_all_runs_wrong_model_audit.json"
    csv_path = out_dir / "chartqa_full_val_all_runs_wrong_model_audit.csv"
    summary_path = out_dir / "chartqa_full_val_all_runs_wrong_model_audit_summary.json"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    fieldnames = [
        "sample_index",
        "human_or_machine",
        "question",
        "reference",
        "primary_error_type",
        "audit_tags",
        "consensus_note",
        "image_path",
        "absolute_image_path",
        *[f"pred_{run}" for run in RUN_FILES],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["audit_tags"] = ";".join(record["audit_tags"])
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if args.contact_sheets:
        write_contact_sheets(records, image_root, out_dir / "contact_sheets")

    print(f"all-runs-wrong records: {len(records)}")
    print(f"wrote: {json_path}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
