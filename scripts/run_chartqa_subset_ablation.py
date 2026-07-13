import argparse
import json
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


RUN_CONFIGS = {
    "baseline_maxpix_802816": {
        "adapter_key": None,
        "max_pixels": 802_816,
        "prompt_name": "default",
    },
    "hardmix_maxpix_602112": {
        "adapter_key": "hardmix",
        "max_pixels": 602_112,
        "prompt_name": "default",
    },
    "hardmix_maxpix_802816": {
        "adapter_key": "hardmix",
        "max_pixels": 802_816,
        "prompt_name": "default",
    },
    "f_maxpix_802816": {
        "adapter_key": "f",
        "max_pixels": 802_816,
        "prompt_name": "default",
    },
    "hardmix_axis_legend_prompt_802816": {
        "adapter_key": "hardmix",
        "max_pixels": 802_816,
        "prompt_name": "axis_legend",
    },
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


def build_prompt(question: str, prompt_name: str) -> str:
    if prompt_name == "default":
        return (
            "Answer the chart question with a concise answer. "
            "If the answer is numeric, return only the number and unit when needed.\n"
            f"Question: {question}"
        )
    if prompt_name == "axis_legend":
        return (
            "You are answering a chart question. First focus on the chart title, axes, tick labels, "
            "legend-color mapping, and visible data labels. Ground the answer in the relevant series "
            "or category before computing. Return only the final concise answer.\n"
            f"Question: {question}"
        )
    raise ValueError(f"Unsupported prompt_name: {prompt_name}")


def predict_with_prompt(
    image: Image.Image,
    question: str,
    prompt_name: str,
    model: Any,
    processor: Any,
    inference_config: InferenceConfig,
    generation_config: GenerationConfig,
    image_path: str,
) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(question, prompt_name)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
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

    return {
        "question": question,
        "answer": answer,
        "model_id": inference_config.model_id,
        "adapter_path": inference_config.adapter_path,
        "latency_seconds": round(latency, 4),
        "generation": asdict(generation_config),
        "image_path": image_path,
    }


def copy_outputs(paths: list[Path], drive_dir: Path | None) -> None:
    if not drive_dir:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, drive_dir / path.name)
            print(f"Copied to Drive: {drive_dir / path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run high-resolution ChartQA all-wrong subset ablations.")
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chartqa_all_wrong_diagnostics"))
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hardmix-adapter-path", type=Path, default=None)
    parser.add_argument("--f-adapter-path", type=Path, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runs", nargs="+", default=list(RUN_CONFIGS))
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, run_name: str, records: list[dict[str, Any]]) -> None:
    if run_name not in RUN_CONFIGS:
        raise ValueError(f"Unknown run {run_name!r}. Choices: {sorted(RUN_CONFIGS)}")
    run_config = RUN_CONFIGS[run_name]

    adapter_paths = {
        "hardmix": args.hardmix_adapter_path,
        "f": args.f_adapter_path,
    }
    adapter_key = run_config["adapter_key"]
    adapter_path = adapter_paths.get(adapter_key) if adapter_key else None
    if adapter_key and not adapter_path:
        raise ValueError(f"Run {run_name} requires --{adapter_key}-adapter-path")
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter for {run_name}: {adapter_path}")

    pred_path = args.output_dir / "predictions" / f"{run_name}.jsonl"
    metrics_path = args.output_dir / "metrics" / f"{run_name}_metrics.json"
    evaluated_path = args.output_dir / "evaluated" / f"{run_name}_evaluated.jsonl"
    errors_path = args.output_dir / "errors" / f"{run_name}_errors.jsonl"

    if args.force_rerun and pred_path.exists():
        pred_path.unlink()

    existing_rows = read_jsonl(pred_path)
    done_indices = {int(row["sample_index"]) for row in existing_rows}
    pending = [row for row in records if int(row["sample_index"]) not in done_indices]

    print(f"\n=== {run_name} ===")
    print(f"adapter_path: {adapter_path or 'baseline'}")
    print(f"max_pixels: {run_config['max_pixels']}")
    print(f"prompt_name: {run_config['prompt_name']}")
    print(f"existing rows: {len(existing_rows)}")
    print(f"pending rows: {len(pending)}")

    if pending:
        inference_config = InferenceConfig(
            model_id=args.model_id,
            adapter_path=str(adapter_path) if adapter_path else None,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            min_pixels=args.min_pixels,
            max_pixels=int(run_config["max_pixels"]),
        )
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

        print("Loading model and processor...")
        model, processor = load_model_and_processor(inference_config)
        model.eval()
        print("Model loaded.")

        for row in tqdm(pending, desc=f"Predicting {run_name}", unit="samples"):
            image_path = Path(row["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for sample {row['sample_index']}: {image_path}")
            image = Image.open(image_path).convert("RGB")
            result = predict_with_prompt(
                image=image,
                question=row["question"],
                prompt_name=str(run_config["prompt_name"]),
                model=model,
                processor=processor,
                inference_config=inference_config,
                generation_config=generation_config,
                image_path=str(image_path),
            )
            output = {
                **result,
                "sample_index": int(row["sample_index"]),
                "selected_index": int(row["sample_index"]),
                "reference_answer": row["reference_answer"],
                "all_labels": row.get("all_labels", [row["reference_answer"]]),
                "human_or_machine": row.get("human_or_machine"),
                "split": "chartqa_all_wrong_diagnostic_subset",
                "run_name": run_name,
                "prompt_name": run_config["prompt_name"],
                "prompt_text": build_prompt(row["question"], str(run_config["prompt_name"])),
                "min_pixels": args.min_pixels,
                "max_pixels": int(run_config["max_pixels"]),
                "load_in_4bit": args.load_in_4bit,
                "reviewed_primary": row.get("reviewed_primary"),
                "review_flags": row.get("review_flags", []),
                "issue_note": row.get("issue_note", ""),
            }
            append_jsonl(pred_path, output)

    rows = read_jsonl(pred_path)
    if len(rows) != len(records):
        raise RuntimeError(f"{run_name} incomplete: {len(rows)}/{len(records)} rows")

    metrics, evaluated, errors = evaluate_records(rows, EvaluationConfig())
    write_json(metrics_path, metrics)
    write_jsonl(evaluated_path, evaluated)
    write_jsonl(errors_path, errors)
    print(f"Relaxed accuracy: {metrics['relaxed_correct']}/{metrics['total']} = {metrics['relaxed_accuracy']:.2%}")

    drive_run_dir = args.drive_output_dir / run_name if args.drive_output_dir else None
    copy_outputs([pred_path, metrics_path, evaluated_path, errors_path], drive_run_dir)


def main() -> int:
    args = parse_args()
    print("subset_jsonl:", args.subset_jsonl)
    print("output_dir:", args.output_dir)
    print("drive_output_dir:", args.drive_output_dir or "skipped")
    print("runs:", args.runs)

    if args.dry_run:
        print("Dry run OK.")
        return 0
    if not args.subset_jsonl.exists():
        raise FileNotFoundError(f"Missing subset JSONL: {args.subset_jsonl}")

    records = read_jsonl(args.subset_jsonl)
    if not records:
        raise ValueError(f"No records found in {args.subset_jsonl}")

    for run_name in args.runs:
        run_one(args, run_name, records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
