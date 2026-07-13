import argparse
import csv
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_chartqa import EvaluationConfig, evaluate_records, write_json, write_jsonl  # noqa: E402
from src.infer import DEFAULT_MODEL_ID, GenerationConfig, InferenceConfig, load_model_and_processor  # noqa: E402


STAGE_PROMPTS = {
    "overview": """You are preparing a chart for reliable question answering.

Look at the chart image and the question, but do not answer the question yet.
Return only valid JSON with:
- chart_type
- title
- subtitle_or_source
- visible_units
- chart_layout
- likely_relevant_regions
- notes

Question: {question}""",
    "axes_legend": """Extract only axis, scale, tick, legend, and color/category mapping information needed for the question.

Use the prior overview for context. Do not answer the question yet.
Return only valid JSON with:
- x_axis: label, tick_labels, date_or_category_order, scale_notes
- y_axis: label, tick_labels, units, min, max, scale_notes
- legend: list of series/category/color mappings
- relevant_visual_elements
- ambiguity_notes

Question: {question}

Prior overview:
{overview}""",
    "data_table": """Extract a question-relevant table from the chart.

Use the overview plus axis/legend extraction. Do not answer the question yet.
Return only valid JSON with:
- table_schema
- data_points: list of rows with series/category/date/value/color/position when visible
- candidate_values_for_question
- missing_or_uncertain_values
- arithmetic_needed

If exact values are not printed, provide the best visually grounded approximate values and mark them as approximate.

Question: {question}

Prior overview:
{overview}

Prior axes and legend extraction:
{axes_legend}""",
}


QA_PROMPTS = {
    "staged_table_json_only": """Answer the ChartQA question using only the staged chart extraction below.

Use the extracted table, axes, legend, and candidate values. If arithmetic is needed, compute carefully. Return only the final concise answer.

Question: {question}

Overview JSON:
{overview}

Axes/legend JSON:
{axes_legend}

Data table JSON:
{data_table}""",
    "staged_image_plus_table_json": """Answer the ChartQA question using both the chart image and the staged chart extraction below.

Use the extraction for candidate values, axes, dates, and legend-color mapping, but verify against the image if the extraction looks incomplete. If arithmetic is needed, compute carefully. Return only the final concise answer.

Question: {question}

Overview JSON:
{overview}

Axes/legend JSON:
{axes_legend}

Data table JSON:
{data_table}""",
}


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def restore_from_drive(local_path: Path, drive_path: Path | None) -> None:
    if local_path.exists() or drive_path is None or not drive_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive_path, local_path)
    print(f"Restored from Drive: {drive_path} -> {local_path}")


def append_with_drive(local_path: Path, drive_path: Path | None, row: dict[str, Any]) -> None:
    append_jsonl(local_path, row)
    if drive_path is not None:
        append_jsonl(drive_path, row)


def copy_outputs(paths: list[Path], drive_dir: Path | None) -> None:
    if not drive_dir:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, drive_dir / path.name)
            print(f"Copied to Drive: {drive_dir / path.name}")


def load_exclude_indices(path: Path | None) -> set[int]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Missing exclude list: {path}")
    indices: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("sample_index"):
                indices.add(int(row["sample_index"]))
    return indices


def best_effort_json_status(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        return {"is_valid_json": False, "json_error": str(exc)}
    return {"is_valid_json": True, "json_keys": sorted(parsed) if isinstance(parsed, dict) else []}


def generate_response(
    messages: list[dict[str, Any]],
    model: Any,
    processor: Any,
    generation_config: GenerationConfig,
) -> tuple[str, float]:
    import torch

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    has_vision = any(
        item.get("type") in {"image", "video"}
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict)
    )
    if has_vision:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **asdict(generation_config))
    latency = time.perf_counter() - start

    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    answer = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer, round(latency, 4)


def stage_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f"{stage}.jsonl"


def drive_stage_path(drive_output_dir: Path | None, stage: str) -> Path | None:
    return drive_output_dir / f"{stage}.jsonl" if drive_output_dir else None


def run_stage(
    *,
    stage: str,
    records: list[dict[str, Any]],
    prior_by_stage: dict[str, dict[int, dict[str, Any]]],
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    inference_config: InferenceConfig,
    generation_config: GenerationConfig,
) -> Path:
    local_path = stage_path(args.output_dir, stage)
    drive_path = drive_stage_path(args.drive_output_dir, stage)
    restore_from_drive(local_path, drive_path)

    existing = read_jsonl(local_path)
    done = {int(row["sample_index"]) for row in existing}
    pending = [row for row in records if int(row["sample_index"]) not in done]
    print(f"\nStage {stage}: existing={len(existing)} pending={len(pending)}")

    for row in tqdm(pending, desc=f"Stage {stage}", unit="samples"):
        sample_index = int(row["sample_index"])
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for sample {sample_index}: {image_path}")
        image = Image.open(image_path).convert("RGB")

        prompt = STAGE_PROMPTS[stage].format(
            question=row["question"],
            overview=prior_by_stage.get("overview", {}).get(sample_index, {}).get("stage_output", ""),
            axes_legend=prior_by_stage.get("axes_legend", {}).get(sample_index, {}).get("stage_output", ""),
        )
        response, latency = generate_response(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            model=model,
            processor=processor,
            generation_config=generation_config,
        )
        output = {
            "sample_index": sample_index,
            "stage": stage,
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "image_path": str(image_path),
            "stage_prompt": prompt,
            "stage_output": response,
            "latency_seconds": latency,
            "json_status": best_effort_json_status(response),
            "model_id": inference_config.model_id,
            "adapter_path": inference_config.adapter_path,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "generation": asdict(generation_config),
            "reviewed_primary": row.get("reviewed_primary"),
            "review_flags": row.get("review_flags", []),
        }
        append_with_drive(local_path, drive_path, output)

    rows = read_jsonl(local_path)
    if len(rows) != len(records):
        raise RuntimeError(f"Stage {stage} incomplete: {len(rows)}/{len(records)} rows")
    return local_path


def run_qa(
    *,
    mode: str,
    records: list[dict[str, Any]],
    stage_outputs: dict[str, dict[int, dict[str, Any]]],
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    inference_config: InferenceConfig,
    generation_config: GenerationConfig,
) -> list[Path]:
    local_path = stage_path(args.output_dir, mode)
    drive_path = drive_stage_path(args.drive_output_dir, mode)
    restore_from_drive(local_path, drive_path)

    existing = read_jsonl(local_path)
    done = {int(row["sample_index"]) for row in existing}
    pending = [row for row in records if int(row["sample_index"]) not in done]
    print(f"\nQA {mode}: existing={len(existing)} pending={len(pending)}")

    for row in tqdm(pending, desc=f"QA {mode}", unit="samples"):
        sample_index = int(row["sample_index"])
        prompt = QA_PROMPTS[mode].format(
            question=row["question"],
            overview=stage_outputs["overview"][sample_index]["stage_output"],
            axes_legend=stage_outputs["axes_legend"][sample_index]["stage_output"],
            data_table=stage_outputs["data_table"][sample_index]["stage_output"],
        )
        if mode == "staged_image_plus_table_json":
            image_path = Path(row["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for sample {sample_index}: {image_path}")
            image = Image.open(image_path).convert("RGB")
            content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]

        answer, latency = generate_response(
            messages=[{"role": "user", "content": content}],
            model=model,
            processor=processor,
            generation_config=generation_config,
        )
        output = {
            "sample_index": sample_index,
            "selected_index": sample_index,
            "question": row["question"],
            "answer": answer,
            "reference_answer": row["reference_answer"],
            "all_labels": row.get("all_labels", [row["reference_answer"]]),
            "human_or_machine": row.get("human_or_machine"),
            "split": "chartqa_all_wrong_diagnostic_subset_valid77",
            "image_path": row["image_path"] if mode == "staged_image_plus_table_json" else None,
            "run_name": mode,
            "prompt_name": mode,
            "prompt_text": prompt,
            "overview_output": stage_outputs["overview"][sample_index]["stage_output"],
            "axes_legend_output": stage_outputs["axes_legend"][sample_index]["stage_output"],
            "data_table_output": stage_outputs["data_table"][sample_index]["stage_output"],
            "latency_seconds": latency,
            "model_id": inference_config.model_id,
            "adapter_path": inference_config.adapter_path,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "load_in_4bit": args.load_in_4bit,
            "generation": asdict(generation_config),
            "reviewed_primary": row.get("reviewed_primary"),
            "review_flags": row.get("review_flags", []),
            "issue_note": row.get("issue_note", ""),
        }
        append_with_drive(local_path, drive_path, output)

    rows = read_jsonl(local_path)
    if len(rows) != len(records):
        raise RuntimeError(f"QA {mode} incomplete: {len(rows)}/{len(records)} rows")

    metrics, evaluated, errors = evaluate_records(rows, EvaluationConfig())
    metrics_path = args.output_dir / f"{mode}_metrics.json"
    evaluated_path = args.output_dir / f"{mode}_evaluated.jsonl"
    errors_path = args.output_dir / f"{mode}_errors.jsonl"
    write_json(metrics_path, metrics)
    write_jsonl(evaluated_path, evaluated)
    write_jsonl(errors_path, errors)
    copy_outputs([metrics_path, evaluated_path, errors_path], args.drive_output_dir)
    print(f"{mode} relaxed: {metrics['relaxed_correct']}/{metrics['total']} = {metrics['relaxed_accuracy']:.2%}")
    return [local_path, metrics_path, evaluated_path, errors_path]


def summarize(
    *,
    records: list[dict[str, Any]],
    stage_outputs: dict[str, dict[int, dict[str, Any]]],
    qa_evaluated_paths: list[Path],
    args: argparse.Namespace,
) -> list[Path]:
    stage_valid_json = {
        stage: sum(1 for row in rows.values() if row.get("json_status", {}).get("is_valid_json"))
        for stage, rows in stage_outputs.items()
    }
    run_summaries = []
    union_correct: set[int] = set()
    for path in qa_evaluated_paths:
        rows = read_jsonl(path)
        run_name = path.name.removesuffix("_evaluated.jsonl")
        correct = {int(row["sample_index"]) for row in rows if row.get("eval_relaxed_correct")}
        union_correct |= correct
        by_primary: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            key = row.get("reviewed_primary", "unknown")
            by_primary[key]["total"] += 1
            by_primary[key]["relaxed_correct"] += int(bool(row.get("eval_relaxed_correct")))
        run_summaries.append(
            {
                "run_name": run_name,
                "total": len(rows),
                "relaxed_correct": len(correct),
                "relaxed_accuracy": len(correct) / len(rows) if rows else 0.0,
                "recovered_indices": sorted(correct),
                "by_reviewed_primary": {key: dict(value) for key, value in by_primary.items()},
            }
        )

    summary = {
        "subset_total_after_excluding_reference_issues": len(records),
        "skipped_exclude_or_fix_reference_count": len(args.exclude_indices),
        "skipped_exclude_or_fix_reference_indices": sorted(args.exclude_indices),
        "stage_valid_json": stage_valid_json,
        "runs": run_summaries,
        "oracle_recovered_count": len(union_correct),
        "oracle_recovered_accuracy": len(union_correct) / len(records) if records else 0.0,
        "oracle_recovered_indices": sorted(union_correct),
        "still_wrong_count": len(records) - len(union_correct),
        "still_wrong_indices": sorted({int(row["sample_index"]) for row in records} - union_correct),
    }

    summary_path = args.output_dir / "staged_extraction_summary.json"
    report_path = args.output_dir / "staged_extraction_report.md"
    write_json(summary_path, summary)
    report_path.write_text(build_report(summary), encoding="utf-8")
    copy_outputs([summary_path, report_path], args.drive_output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return [summary_path, report_path]


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Module 22B staged chart-to-table extraction report",
        "",
        f"Valid subset size after excluding reference issues: {summary['subset_total_after_excluding_reference_issues']}",
        f"Skipped exclude/fix-reference samples: {summary['skipped_exclude_or_fix_reference_indices']}",
        "",
        "## Stage JSON validity",
        "",
        "| stage | valid JSON |",
        "|---|---:|",
    ]
    for stage, count in summary["stage_valid_json"].items():
        lines.append(f"| `{stage}` | {count} |")

    lines.extend(["", "## QA runs", "", "| run | relaxed |", "|---|---:|"])
    for run in summary["runs"]:
        lines.append(
            f"| `{run['run_name']}` | {run['relaxed_correct']}/{run['total']} = {run['relaxed_accuracy']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Oracle",
            "",
            f"- oracle recovered: {summary['oracle_recovered_count']}/{summary['subset_total_after_excluding_reference_issues']} = {summary['oracle_recovered_accuracy']:.2%}",
            f"- still wrong: {summary['still_wrong_count']}",
            f"- recovered indices: {summary['oracle_recovered_indices']}",
            f"- still wrong indices: {summary['still_wrong_indices']}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged chart-to-table extraction diagnostics for Module 22B.")
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--exclude-list-csv", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/chartqa_all_wrong_diagnostics/staged_extraction"),
    )
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-pixels", type=int, default=802_816)
    parser.add_argument("--stage-max-new-tokens", type=int, default=768)
    parser.add_argument("--qa-max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("subset_jsonl:", args.subset_jsonl)
    print("exclude_list_csv:", args.exclude_list_csv or "skipped")
    print("output_dir:", args.output_dir)
    print("drive_output_dir:", args.drive_output_dir or "skipped")

    if args.dry_run:
        print("Dry run OK.")
        return 0
    if not args.subset_jsonl.exists():
        raise FileNotFoundError(f"Missing subset JSONL: {args.subset_jsonl}")
    if args.adapter_path and not args.adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {args.adapter_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.drive_output_dir:
        args.drive_output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_rerun:
        for path in list(args.output_dir.glob("*.jsonl")) + list(args.output_dir.glob("*.json")) + list(args.output_dir.glob("*.md")):
            path.unlink()
        if args.drive_output_dir:
            for path in list(args.drive_output_dir.glob("*.jsonl")) + list(args.drive_output_dir.glob("*.json")) + list(args.drive_output_dir.glob("*.md")):
                path.unlink()

    records = read_jsonl(args.subset_jsonl)
    exclude_indices = load_exclude_indices(args.exclude_list_csv)
    args.exclude_indices = exclude_indices
    if exclude_indices:
        records = [row for row in records if int(row["sample_index"]) not in exclude_indices]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No records left after filtering")

    print(f"Records to process: {len(records)}")
    print(f"Excluded reference/evaluator issue samples: {sorted(exclude_indices)}")

    inference_config = InferenceConfig(
        model_id=args.model_id,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    print("Loading model and processor...")
    model, processor = load_model_and_processor(inference_config)
    model.eval()
    print("Model loaded.")

    stage_generation = GenerationConfig(
        max_new_tokens=args.stage_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    qa_generation = GenerationConfig(
        max_new_tokens=args.qa_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    stage_outputs: dict[str, dict[int, dict[str, Any]]] = {}
    for stage in ["overview", "axes_legend", "data_table"]:
        run_stage(
            stage=stage,
            records=records,
            prior_by_stage=stage_outputs,
            args=args,
            model=model,
            processor=processor,
            inference_config=inference_config,
            generation_config=stage_generation,
        )
        stage_outputs[stage] = {
            int(row["sample_index"]): row
            for row in read_jsonl(stage_path(args.output_dir, stage))
        }

    qa_evaluated_paths = []
    for mode in ["staged_table_json_only", "staged_image_plus_table_json"]:
        paths = run_qa(
            mode=mode,
            records=records,
            stage_outputs=stage_outputs,
            args=args,
            model=model,
            processor=processor,
            inference_config=inference_config,
            generation_config=qa_generation,
        )
        qa_evaluated_paths.extend([path for path in paths if path.name.endswith("_evaluated.jsonl")])

    summarize(records=records, stage_outputs=stage_outputs, qa_evaluated_paths=qa_evaluated_paths, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
