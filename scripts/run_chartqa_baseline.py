import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_chartqa import exact_match  # noqa: E402
from src.infer import (  # noqa: E402
    DEFAULT_MODEL_ID,
    GenerationConfig,
    InferenceConfig,
    load_model_and_processor,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL baseline on a small ChartQA stream.")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-pixels", type=int, default=401_408)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without loading model or dataset.",
    )
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    return Path(f"outputs/chartqa_baseline_{args.n_samples}.jsonl")


def main() -> int:
    args = parse_args()
    output_path = resolve_output_path(args)

    # baseline 只验证推理闭环，不引入训练逻辑；后续 QLoRA 仍复用这套模型参数。
    inference_config = InferenceConfig(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
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

    if args.dry_run:
        print("Dry run OK.")
        print("n_samples:", args.n_samples)
        print("split:", args.split)
        print("output:", output_path)
        print("drive_output_dir:", args.drive_output_dir)
        print("inference_config:", asdict(inference_config))
        print("generation_config:", asdict(generation_config))
        return 0

    import torch
    from datasets import load_dataset

    # 真实 baseline 必须在 Colab GPU 上运行；CPU runtime 即使能下载模型，也没有工程价值。
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Switch Colab runtime to GPU before running baseline.")

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))

    print("\nLoading model and processor...")
    model, processor = load_model_and_processor(inference_config)
    model.eval()
    print("Model loaded.")

    print(f"\nLoading ChartQA streaming dataset split={args.split!r}...")
    # 小批量测试用 streaming，避免为了 5/20/100 条样本下载完整数据集。
    dataset = load_dataset("HuggingFaceM4/ChartQA", split=args.split, streaming=True)

    records: list[dict[str, Any]] = []
    start_all = time.perf_counter()

    for idx, sample in enumerate(dataset):
        if idx >= args.n_samples:
            break

        image = sample["image"].convert("RGB")
        question = sample["query"]
        labels = sample["label"]
        # ChartQA 的 label 通常是列表；baseline 阶段用第一个答案做即时反馈。
        reference_answer = labels[0] if isinstance(labels, list) and labels else labels

        print(f"\n[{idx + 1}/{args.n_samples}] Q: {question}")
        print("Reference:", reference_answer)

        result = predict(
            image=image,
            question=question,
            model=model,
            processor=processor,
            inference_config=inference_config,
            generation_config=generation_config,
            image_path=None,
        )

        # 这里的 exact 只用于现场观察；正式报告用 evaluate_predictions.py 重新算 relaxed 指标。
        is_exact = exact_match(result.answer, reference_answer)
        record = asdict(result)
        record.update(
            {
                "sample_index": idx,
                "reference_answer": reference_answer,
                "all_labels": labels,
                "human_or_machine": sample.get("human_or_machine"),
                "split": f"{args.split}_streaming_head",
                "exact_match": is_exact,
            }
        )
        records.append(record)

        print("Prediction:", result.answer)
        print("Exact match:", is_exact)
        print("Latency seconds:", result.latency_seconds)

    elapsed = time.perf_counter() - start_all
    exact_count = sum(1 for record in records if record["exact_match"])
    exact_acc = exact_count / len(records) if records else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} predictions to {output_path}")
    print(f"Exact match: {exact_count}/{len(records)} = {exact_acc:.2%}")
    print(f"Total elapsed seconds: {elapsed:.2f}")

    if args.drive_output_dir:
        args.drive_output_dir.mkdir(parents=True, exist_ok=True)
        drive_output_path = args.drive_output_dir / output_path.name
        # copy2 写入固定文件名；重复运行同一个 RUN_NAME 会覆盖，不会在 Drive 中无限累积。
        shutil.copy2(output_path, drive_output_path)
        print(f"Copied output to Drive: {drive_output_path}")

    print("\nPreview:")
    for record in records:
        print(
            {
                "idx": record["sample_index"],
                "question": record["question"],
                "prediction": record["answer"],
                "reference": record["reference_answer"],
                "exact_match": record["exact_match"],
                "latency": record["latency_seconds"],
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

