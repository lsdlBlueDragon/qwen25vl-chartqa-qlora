import argparse
import json
import re
import shutil
import sys
import time
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


EXTRACTION_PROMPT = """Extract the chart into structured JSON for downstream question answering.

Return only valid JSON with these keys:
- chart_type
- title
- x_axis
- y_axis
- legend
- visible_text
- data_points
- notes

For data_points, include series/category/date/value/color when visible. Preserve approximate values if exact values are not printed. Do not answer a specific question yet."""


def qa_prompt(question: str, extraction: str, mode: str) -> str:
    if mode == "table_json_only":
        return (
            "Answer the ChartQA question using only the structured chart extraction below. "
            "If arithmetic is needed, compute carefully from the extracted values. "
            "Return only the final concise answer.\n\n"
            f"Structured chart extraction:\n{extraction}\n\n"
            f"Question: {question}"
        )
    if mode == "image_plus_table_json":
        return (
            "Answer the ChartQA question using both the chart image and the structured extraction. "
            "Use the extraction for values, axes, dates, and legend-color mapping, but verify against the image when needed. "
            "If arithmetic is needed, compute carefully. Return only the final concise answer.\n\n"
            f"Structured chart extraction:\n{extraction}\n\n"
            f"Question: {question}"
        )
    raise ValueError(f"Unsupported QA mode: {mode}")


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


def copy_outputs(paths: list[Path], drive_dir: Path | None) -> None:
    if not drive_dir:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, drive_dir / path.name)
            print(f"Copied to Drive: {drive_dir / path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured extraction diagnostics on ChartQA all-wrong subset.")
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chartqa_all_wrong_diagnostics"))
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-pixels", type=int, default=802_816)
    parser.add_argument("--extract-max-new-tokens", type=int, default=768)
    parser.add_argument("--qa-max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_extraction(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    model: Any,
    processor: Any,
    inference_config: InferenceConfig,
) -> Path:
    extraction_path = args.output_dir / "extractions" / "qwen25vl3b_chart_to_json_802816.jsonl"
    if args.force_rerun and extraction_path.exists():
        extraction_path.unlink()

    existing = read_jsonl(extraction_path)
    done = {int(row["sample_index"]) for row in existing}
    pending = [row for row in records if int(row["sample_index"]) not in done]
    print(f"Extraction existing rows: {len(existing)}")
    print(f"Extraction pending rows: {len(pending)}")

    generation_config = GenerationConfig(
        max_new_tokens=args.extract_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    for row in tqdm(pending, desc="Chart-to-JSON extraction", unit="samples"):
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for sample {row['sample_index']}: {image_path}")
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]
        extraction, latency = generate_response(messages, model, processor, generation_config)
        append_jsonl(
            extraction_path,
            {
                "sample_index": int(row["sample_index"]),
                "image_path": str(image_path),
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "extraction": extraction,
                "latency_seconds": latency,
                "model_id": inference_config.model_id,
                "adapter_path": inference_config.adapter_path,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "generation": asdict(generation_config),
                "json_status": best_effort_json_status(extraction),
                "reviewed_primary": row.get("reviewed_primary"),
                "review_flags": row.get("review_flags", []),
            },
        )

    rows = read_jsonl(extraction_path)
    if len(rows) != len(records):
        raise RuntimeError(f"Extraction incomplete: {len(rows)}/{len(records)} rows")
    return extraction_path


def run_qa_mode(
    args: argparse.Namespace,
    mode: str,
    records: list[dict[str, Any]],
    extractions_by_index: dict[int, dict[str, Any]],
    model: Any,
    processor: Any,
    inference_config: InferenceConfig,
) -> list[Path]:
    pred_path = args.output_dir / "table_qa" / f"{mode}.jsonl"
    metrics_path = args.output_dir / "table_qa_metrics" / f"{mode}_metrics.json"
    evaluated_path = args.output_dir / "table_qa_evaluated" / f"{mode}_evaluated.jsonl"
    errors_path = args.output_dir / "table_qa_errors" / f"{mode}_errors.jsonl"

    if args.force_rerun and pred_path.exists():
        pred_path.unlink()

    existing = read_jsonl(pred_path)
    done = {int(row["sample_index"]) for row in existing}
    pending = [row for row in records if int(row["sample_index"]) not in done]
    print(f"{mode} existing rows: {len(existing)}")
    print(f"{mode} pending rows: {len(pending)}")

    generation_config = GenerationConfig(
        max_new_tokens=args.qa_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )

    for row in tqdm(pending, desc=f"QA {mode}", unit="samples"):
        sample_index = int(row["sample_index"])
        extraction = extractions_by_index[sample_index]["extraction"]
        prompt = qa_prompt(row["question"], extraction, mode)
        if mode == "image_plus_table_json":
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
        append_jsonl(
            pred_path,
            {
                "sample_index": sample_index,
                "selected_index": sample_index,
                "question": row["question"],
                "answer": answer,
                "reference_answer": row["reference_answer"],
                "all_labels": row.get("all_labels", [row["reference_answer"]]),
                "human_or_machine": row.get("human_or_machine"),
                "split": "chartqa_all_wrong_diagnostic_subset",
                "image_path": row["image_path"] if mode == "image_plus_table_json" else None,
                "run_name": mode,
                "prompt_name": mode,
                "prompt_text": prompt,
                "structured_extraction": extraction,
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
            },
        )

    rows = read_jsonl(pred_path)
    if len(rows) != len(records):
        raise RuntimeError(f"{mode} incomplete: {len(rows)}/{len(records)} rows")
    metrics, evaluated, errors = evaluate_records(rows, EvaluationConfig())
    write_json(metrics_path, metrics)
    write_jsonl(evaluated_path, evaluated)
    write_jsonl(errors_path, errors)
    print(f"{mode} relaxed accuracy: {metrics['relaxed_correct']}/{metrics['total']} = {metrics['relaxed_accuracy']:.2%}")
    return [pred_path, metrics_path, evaluated_path, errors_path]


def main() -> int:
    args = parse_args()
    print("subset_jsonl:", args.subset_jsonl)
    print("output_dir:", args.output_dir)
    print("drive_output_dir:", args.drive_output_dir or "skipped")
    print("adapter_path:", args.adapter_path or "baseline")

    if args.dry_run:
        print("Dry run OK.")
        return 0
    if not args.subset_jsonl.exists():
        raise FileNotFoundError(f"Missing subset JSONL: {args.subset_jsonl}")
    if args.adapter_path and not args.adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter: {args.adapter_path}")

    records = read_jsonl(args.subset_jsonl)
    if not records:
        raise ValueError(f"No records found in {args.subset_jsonl}")

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

    extraction_path = run_extraction(args, records, model, processor, inference_config)
    extractions = read_jsonl(extraction_path)
    extractions_by_index = {int(row["sample_index"]): row for row in extractions}

    output_paths = [extraction_path]
    output_paths.extend(
        run_qa_mode(args, "table_json_only", records, extractions_by_index, model, processor, inference_config)
    )
    output_paths.extend(
        run_qa_mode(args, "image_plus_table_json", records, extractions_by_index, model, processor, inference_config)
    )

    if args.drive_output_dir:
        copy_outputs(output_paths, args.drive_output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
