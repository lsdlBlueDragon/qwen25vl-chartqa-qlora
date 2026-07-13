#!/usr/bin/env python
"""Module 23C: targeted prompt/evaluator ablation for hard ChartQA failures.

This script is intended for Colab/GPU. It runs only the Module 23C true-hard
subset, supports append-only checkpoints, Drive restore/sync, and evaluates
outputs with both the base ChartQA evaluator and Module 23C normalization v2.
"""

from __future__ import annotations

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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [PROJECT_ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.eval_chartqa import EvaluationConfig, classify_record, write_json, write_jsonl  # noqa: E402
from src.infer import DEFAULT_MODEL_ID, GenerationConfig, InferenceConfig, load_model_and_processor  # noqa: E402
from run_chartqa_23c_normalization_v2 import normalization_v2_match  # noqa: E402


PROMPTS = {
    "legend_table_prompt": (
        "Answer the chart question. Before answering, identify the chart's legend, colors, "
        "series names, and any color-to-label mapping relevant to the question. Then return "
        "only the final concise answer.\nQuestion: {question}"
    ),
    "operand_table_prompt": (
        "Answer the chart question. First extract the exact operands needed for the requested "
        "calculation as a compact table or list. Then compute the result. Return only the final "
        "concise answer.\nQuestion: {question}"
    ),
    "spatial_locator_prompt": (
        "Answer the chart question. If the question uses positional words such as left, right, "
        "top, bottom, middle, first, or last, locate the requested row/bar/segment explicitly "
        "before answering. Return only the final concise answer.\nQuestion: {question}"
    ),
    "range_count_prompt": (
        "Answer the chart question. If it asks about a range, threshold, or count, enumerate the "
        "relevant categories/years that satisfy the condition, then give only the final concise "
        "answer.\nQuestion: {question}"
    ),
    "multi_answer_prompt": (
        "Answer the chart question. If multiple labels, values, or years satisfy the question, "
        "return the complete list and do not stop after the first match. Return only the final "
        "concise answer.\nQuestion: {question}"
    ),
}

BUCKET_TO_PROMPT = {
    "legend_color_mapping": "legend_table_prompt",
    "semantic_category_filtering": "range_count_prompt",
    "timepoint_count_or_threshold_count": "range_count_prompt",
    "date_axis_peak_or_extreme": "multi_answer_prompt",
    "spatial_position_grounding": "spatial_locator_prompt",
    "arithmetic_average_or_median": "operand_table_prompt",
    "arithmetic_sum_or_difference": "operand_table_prompt",
    "boolean_after_computation_or_trend": "operand_table_prompt",
    "label_after_computation": "operand_table_prompt",
    "range_or_threshold_aggregation": "range_count_prompt",
    "ranking_after_difference": "operand_table_prompt",
    "specific_value_lookup_with_axis": "legend_table_prompt",
    "extreme_or_ranking": "spatial_locator_prompt",
    "numeric_value_extraction": "legend_table_prompt",
    "multi_step_reasoning": "operand_table_prompt",
}

ACTION_TO_PROMPT = {
    "legend": "legend_table_prompt",
    "color": "legend_table_prompt",
    "crop": "legend_table_prompt",
    "ocr": "legend_table_prompt",
    "operand": "operand_table_prompt",
    "compute": "operand_table_prompt",
    "arithmetic": "operand_table_prompt",
    "average": "operand_table_prompt",
    "difference": "operand_table_prompt",
    "layout": "spatial_locator_prompt",
    "locator": "spatial_locator_prompt",
    "localization": "spatial_locator_prompt",
    "position": "spatial_locator_prompt",
    "range": "range_count_prompt",
    "threshold": "range_count_prompt",
    "enumerate": "range_count_prompt",
    "count": "range_count_prompt",
    "exhaustive": "multi_answer_prompt",
    "multi": "multi_answer_prompt",
    "list": "multi_answer_prompt",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def restore_from_drive(local_path: Path, drive_path: Path | None) -> None:
    if local_path.exists() or not drive_path or not drive_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive_path, local_path)
    print(f"Restored from Drive: {drive_path} -> {local_path}")


def sync_to_drive(local_paths: list[Path], drive_dir: Path | None) -> None:
    if not drive_dir:
        return
    drive_dir.mkdir(parents=True, exist_ok=True)
    for path in local_paths:
        if path.exists():
            shutil.copy2(path, drive_dir / path.name)


def build_prompt(question: str, prompt_name: str) -> str:
    return PROMPTS[prompt_name].format(question=question)


def prompts_for_sample(row: dict[str, Any], policy: str) -> list[str]:
    if policy == "all":
        return list(PROMPTS)
    if policy == "routed":
        if row.get("target_failure_bucket"):
            return [BUCKET_TO_PROMPT.get(row["target_failure_bucket"], "operand_table_prompt")]
        route_text = f"{row.get('specific_failure', '')} {row.get('target_next_action', '')}".lower()
        for key, prompt_name in ACTION_TO_PROMPT.items():
            if key in route_text:
                return [prompt_name]
        return ["operand_table_prompt"]
    raise ValueError(f"Unsupported prompt policy: {policy}")


def predict_one(
    image: Image.Image,
    prompt_text: str,
    model: Any,
    processor: Any,
    inference_config: InferenceConfig,
    generation_config: GenerationConfig,
) -> tuple[str, float]:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
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
    ).to(model.device)

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


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        base = classify_record(row, EvaluationConfig())
        extra, rule = normalization_v2_match(base)
        final_correct = bool(base["eval_relaxed_correct"] or extra)
        evaluated.append(
            {
                **base,
                "eval_normalization_v2_extra_correct": bool(extra and not base["eval_relaxed_correct"]),
                "eval_normalization_v2_rule": rule if extra and not base["eval_relaxed_correct"] else "",
                "eval_normalization_v2_correct": final_correct,
            }
        )

    total = len(evaluated)
    by_prompt: dict[str, dict[str, Any]] = {}
    for row in evaluated:
        group = by_prompt.setdefault(
            row["prompt_name"],
            {"total": 0, "base_relaxed_correct": 0, "normalization_v2_correct": 0, "recovered_indices": []},
        )
        group["total"] += 1
        group["base_relaxed_correct"] += int(row["eval_relaxed_correct"])
        group["normalization_v2_correct"] += int(row["eval_normalization_v2_correct"])
        if row["eval_normalization_v2_correct"]:
            group["recovered_indices"].append(int(row["sample_index"]))

    for group in by_prompt.values():
        group["base_relaxed_accuracy"] = group["base_relaxed_correct"] / group["total"] if group["total"] else 0.0
        group["normalization_v2_accuracy"] = (
            group["normalization_v2_correct"] / group["total"] if group["total"] else 0.0
        )
        group["recovered_indices"] = sorted(set(group["recovered_indices"]))

    oracle = {int(row["sample_index"]) for row in evaluated if row["eval_normalization_v2_correct"]}
    metrics = {
        "total_predictions": total,
        "unique_samples": len({int(row["sample_index"]) for row in evaluated}),
        "oracle_recovered_count": len(oracle),
        "oracle_recovered_indices": sorted(oracle),
        "by_prompt": by_prompt,
    }
    return evaluated, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--targeted-review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chartqa_23c_targeted_prompt_ablation"))
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--adapter-name", default="baseline_or_adapter")
    parser.add_argument("--prompt-policy", choices=["routed", "all"], default="routed")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-pixels", type=int, default=802_816)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / f"targeted_prompt_{args.adapter_name}_{args.prompt_policy}.jsonl"
    evaluated_path = args.output_dir / f"targeted_prompt_{args.adapter_name}_{args.prompt_policy}_evaluated.jsonl"
    metrics_path = args.output_dir / f"targeted_prompt_{args.adapter_name}_{args.prompt_policy}_metrics.json"

    drive_pred_path = args.drive_output_dir / pred_path.name if args.drive_output_dir else None
    restore_from_drive(pred_path, drive_pred_path)

    subset = {int(row["sample_index"]): row for row in read_jsonl(args.subset_jsonl)}
    review_rows = [
        row for row in read_csv(args.targeted_review_csv) if row["review_group"] == "true_hard_failure"
    ]
    tasks: list[dict[str, Any]] = []
    for review in review_rows:
        idx = int(review["sample_index"])
        if idx not in subset:
            raise KeyError(f"sample_index {idx} is not in subset JSONL")
        row = {**subset[idx], **review}
        for prompt_name in prompts_for_sample(row, args.prompt_policy):
            tasks.append({**row, "prompt_name": prompt_name})

    existing = read_jsonl(pred_path)
    done = {(int(row["sample_index"]), row["prompt_name"]) for row in existing}
    pending = [task for task in tasks if (int(task["sample_index"]), task["prompt_name"]) not in done]

    print("subset_jsonl:", args.subset_jsonl)
    print("targeted_review_csv:", args.targeted_review_csv)
    print("output_dir:", args.output_dir)
    print("drive_output_dir:", args.drive_output_dir or "skipped")
    print("prompt_policy:", args.prompt_policy)
    print("tasks:", len(tasks), "existing:", len(existing), "pending:", len(pending))

    if args.dry_run:
        print("Dry run OK.")
        return 0
    if args.adapter_path and not args.adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter path: {args.adapter_path}")

    if pending:
        inference_config = InferenceConfig(
            model_id=args.model_id,
            adapter_path=str(args.adapter_path) if args.adapter_path else None,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
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

        for i, task in enumerate(tqdm(pending, desc="Targeted prompt ablation", unit="pred"), start=1):
            image_path = Path(task["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            prompt_text = build_prompt(task["question"], task["prompt_name"])
            answer, latency = predict_one(
                image=image,
                prompt_text=prompt_text,
                model=model,
                processor=processor,
                inference_config=inference_config,
                generation_config=generation_config,
            )
            output = {
                "sample_index": int(task["sample_index"]),
                "selected_index": int(task["sample_index"]),
                "question": task["question"],
                "answer": answer,
                "reference_answer": task["reference_answer"],
                "all_labels": task.get("all_labels", [task["reference_answer"]]),
                "human_or_machine": task.get("human_or_machine"),
                "split": "chartqa_23c_true_hard_subset",
                "image_path": str(image_path),
                "run_name": f"targeted_prompt_{args.adapter_name}_{args.prompt_policy}",
                "prompt_name": task["prompt_name"],
                "prompt_text": prompt_text,
                "target_failure_bucket": task.get("target_failure_bucket", ""),
                "specific_failure": task["specific_failure"],
                "target_next_action": task["target_next_action"],
                "model_id": args.model_id,
                "adapter_path": str(args.adapter_path) if args.adapter_path else None,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
                "load_in_4bit": args.load_in_4bit,
                "latency_seconds": latency,
                "generation": asdict(generation_config),
            }
            append_jsonl(pred_path, output)
            if args.sync_every > 0 and i % args.sync_every == 0:
                sync_to_drive([pred_path], args.drive_output_dir)

    rows = read_jsonl(pred_path)
    if len(rows) != len(tasks):
        raise RuntimeError(f"Incomplete predictions: {len(rows)}/{len(tasks)}")
    evaluated, metrics = evaluate_prediction_rows(rows)
    write_jsonl(evaluated_path, evaluated)
    write_json(metrics_path, metrics)
    sync_to_drive([pred_path, evaluated_path, metrics_path], args.drive_output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
