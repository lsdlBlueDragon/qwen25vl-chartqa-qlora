#!/usr/bin/env python
"""Module 24A: structured-intermediate ablation for ChartQA true-hard samples.

This script is intended for Colab/GPU. It runs only the true-hard samples from
Module 23B and asks the model to produce a structured intermediate answer
before the final answer. Outputs are append-only JSONL and can be restored from
Drive after Colab disconnects.
"""

from __future__ import annotations

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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [PROJECT_ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.eval_chartqa import EvaluationConfig, classify_record, write_json, write_jsonl  # noqa: E402
from src.infer import DEFAULT_MODEL_ID, GenerationConfig, InferenceConfig, load_model_and_processor  # noqa: E402
from run_chartqa_23c_normalization_v2 import normalization_v2_match  # noqa: E402


PROMPTS = {
    "legend_mapping_schema": (
        "You are solving a chart QA problem. Return only valid JSON.\n"
        "Use this schema:\n"
        "{\n"
        '  "task_type": "legend_or_color_mapping",\n'
        '  "legend_or_color_map": [{"visual": "...", "label": "..."}],\n'
        '  "relevant_visual": "...",\n'
        '  "evidence": "...",\n'
        '  "final_answer": "..."\n'
        "}\n"
        "The final_answer must be concise and contain only the answer.\n"
        "Question: {question}"
    ),
    "operand_schema": (
        "You are solving a chart QA calculation problem. Return only valid JSON.\n"
        "Use this schema:\n"
        "{\n"
        '  "task_type": "operand_calculation",\n'
        '  "needed_values": [{"label": "...", "value": "..."}],\n'
        '  "operation": "...",\n'
        '  "calculation": "...",\n'
        '  "final_answer": "..."\n'
        "}\n"
        "List the operands first, then compute. The final_answer must be concise.\n"
        "Question: {question}"
    ),
    "spatial_schema": (
        "You are solving a chart QA problem involving spatial or positional language. Return only valid JSON.\n"
        "Use this schema:\n"
        "{\n"
        '  "task_type": "spatial_locator",\n'
        '  "layout_target": {"row": "...", "column_or_segment": "...", "position_phrase": "..."},\n'
        '  "localized_value_or_label": "...",\n'
        '  "evidence": "...",\n'
        '  "final_answer": "..."\n'
        "}\n"
        "The final_answer must be concise and contain only the requested value or label.\n"
        "Question: {question}"
    ),
    "range_count_schema": (
        "You are solving a chart QA range, threshold, or counting problem. Return only valid JSON.\n"
        "Use this schema:\n"
        "{\n"
        '  "task_type": "range_or_count",\n'
        '  "condition": "...",\n'
        '  "items_checked": [{"item": "...", "value": "...", "satisfies": true}],\n'
        '  "count_or_aggregation": "...",\n'
        '  "final_answer": "..."\n'
        "}\n"
        "Enumerate all relevant items before answering. The final_answer must be concise.\n"
        "Question: {question}"
    ),
    "multi_answer_schema": (
        "You are solving a chart QA problem where more than one label, value, or year may be correct. Return only valid JSON.\n"
        "Use this schema:\n"
        "{\n"
        '  "task_type": "multi_answer_or_tie",\n'
        '  "criterion": "...",\n'
        '  "all_matching_items": ["...", "..."],\n'
        '  "evidence": "...",\n'
        '  "final_answer": ["...", "..."]\n'
        "}\n"
        "Return the complete list in final_answer, not only the first match.\n"
        "Question: {question}"
    ),
}

ACTION_TO_PROMPT = {
    "legend": "legend_mapping_schema",
    "color": "legend_mapping_schema",
    "crop": "legend_mapping_schema",
    "ocr": "legend_mapping_schema",
    "operand": "operand_schema",
    "compute": "operand_schema",
    "arithmetic": "operand_schema",
    "average": "operand_schema",
    "difference": "operand_schema",
    "layout": "spatial_schema",
    "locator": "spatial_schema",
    "localization": "spatial_schema",
    "position": "spatial_schema",
    "range": "range_count_schema",
    "threshold": "range_count_schema",
    "enumerate": "range_count_schema",
    "count": "range_count_schema",
    "exhaustive": "multi_answer_schema",
    "multi": "multi_answer_schema",
    "list": "multi_answer_schema",
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


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.S | re.I)
    return match.group(1).strip() if match else stripped


def parse_structured_output(text: str) -> tuple[Any, str | None]:
    try:
        return json.loads(strip_code_fence(text)), None
    except Exception as exc:  # noqa: BLE001 - record parse error in output.
        return None, str(exc)


def final_answer_to_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def prompt_for_sample(row: dict[str, Any], policy: str) -> list[str]:
    if policy == "all":
        return list(PROMPTS)
    if policy != "routed":
        raise ValueError(f"Unsupported prompt policy: {policy}")
    route_text = f"{row.get('specific_failure', '')} {row.get('target_next_action', '')}".lower()
    for key, prompt_name in ACTION_TO_PROMPT.items():
        if key in route_text:
            return [prompt_name]
    return ["operand_schema"]


def build_prompt(question: str, prompt_name: str) -> str:
    return PROMPTS[prompt_name].format(question=question)


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


def evaluate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        parsed, parse_error = parse_structured_output(row["structured_output"])
        final_answer = parsed.get("final_answer") if isinstance(parsed, dict) else None
        answer_for_eval = final_answer_to_text(final_answer) if final_answer is not None else row["structured_output"]
        eval_record = {**row, "answer": answer_for_eval}
        base = classify_record(eval_record, EvaluationConfig())
        extra, rule = normalization_v2_match(base)
        evaluated.append(
            {
                **base,
                "structured_json_valid": parse_error is None,
                "structured_json_error": parse_error or "",
                "parsed_final_answer": answer_for_eval,
                "eval_normalization_v2_extra_correct": bool(extra and not base["eval_relaxed_correct"]),
                "eval_normalization_v2_rule": rule if extra and not base["eval_relaxed_correct"] else "",
                "eval_normalization_v2_correct": bool(base["eval_relaxed_correct"] or extra),
            }
        )

    by_prompt: dict[str, dict[str, Any]] = {}
    for row in evaluated:
        group = by_prompt.setdefault(
            row["prompt_name"],
            {
                "total": 0,
                "valid_json": 0,
                "base_relaxed_correct": 0,
                "normalization_v2_correct": 0,
                "recovered_indices": [],
            },
        )
        group["total"] += 1
        group["valid_json"] += int(row["structured_json_valid"])
        group["base_relaxed_correct"] += int(row["eval_relaxed_correct"])
        group["normalization_v2_correct"] += int(row["eval_normalization_v2_correct"])
        if row["eval_normalization_v2_correct"]:
            group["recovered_indices"].append(int(row["sample_index"]))

    for group in by_prompt.values():
        total = group["total"]
        group["valid_json_rate"] = group["valid_json"] / total if total else 0.0
        group["base_relaxed_accuracy"] = group["base_relaxed_correct"] / total if total else 0.0
        group["normalization_v2_accuracy"] = group["normalization_v2_correct"] / total if total else 0.0
        group["recovered_indices"] = sorted(set(group["recovered_indices"]))

    oracle = {int(row["sample_index"]) for row in evaluated if row["eval_normalization_v2_correct"]}
    metrics = {
        "total_predictions": len(evaluated),
        "unique_samples": len({int(row["sample_index"]) for row in evaluated}),
        "valid_json": sum(int(row["structured_json_valid"]) for row in evaluated),
        "oracle_recovered_count": len(oracle),
        "oracle_recovered_indices": sorted(oracle),
        "by_prompt": by_prompt,
    }
    return evaluated, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--targeted-review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chartqa_24a_structured_hard_ablation"))
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--adapter-name", default="baseline_or_adapter")
    parser.add_argument("--prompt-policy", choices=["routed", "all"], default="routed")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-pixels", type=int, default=802_816)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"structured_24a_{args.adapter_name}_{args.prompt_policy}"
    pred_path = args.output_dir / f"{run_stem}.jsonl"
    evaluated_path = args.output_dir / f"{run_stem}_evaluated.jsonl"
    metrics_path = args.output_dir / f"{run_stem}_metrics.json"

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
            raise KeyError(f"sample_index {idx} missing from subset")
        row = {**subset[idx], **review}
        for prompt_name in prompt_for_sample(row, args.prompt_policy):
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

        for i, task in enumerate(tqdm(pending, desc="24A structured ablation", unit="pred"), start=1):
            image_path = Path(task["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            prompt_text = build_prompt(task["question"], task["prompt_name"])
            structured_output, latency = predict_one(
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
                "answer": structured_output,
                "structured_output": structured_output,
                "reference_answer": task["reference_answer"],
                "all_labels": task.get("all_labels", [task["reference_answer"]]),
                "human_or_machine": task.get("human_or_machine"),
                "split": "chartqa_24a_true_hard_subset",
                "image_path": str(image_path),
                "run_name": run_stem,
                "prompt_name": task["prompt_name"],
                "prompt_text": prompt_text,
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
    evaluated, metrics = evaluate_rows(rows)
    write_jsonl(evaluated_path, evaluated)
    write_json(metrics_path, metrics)
    sync_to_drive([pred_path, evaluated_path, metrics_path], args.drive_output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
