import argparse
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import (
    DEFAULT_MODEL_ID,
    GenerationConfig,
    InferenceConfig,
    load_image,
    load_model_and_processor,
    predict,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL baseline inference on one chart image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to a chart image.")
    parser.add_argument("--question", required=True, help="Chart question.")
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_single.jsonl"))
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
        help="Validate arguments and write no prediction. Does not import model dependencies.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 单图脚本是最小推理闭环；批量 ChartQA baseline 会复用同一套推理配置。
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
        print("Image:", args.image)
        print("Question:", args.question)
        print("Output:", args.output)
        print("Inference config:", asdict(inference_config))
        print("Generation config:", asdict(generation_config))
        return 0

    # 非 dry-run 会加载 3B 权重，默认应在 Colab GPU 或高显存环境运行。
    image = load_image(args.image)
    model, processor = load_model_and_processor(inference_config)
    result = predict(
        image=image,
        question=args.question,
        model=model,
        processor=processor,
        inference_config=inference_config,
        generation_config=generation_config,
        image_path=str(args.image),
    )
    write_jsonl(args.output, [asdict(result)])
    print(result.answer)
    print(f"Wrote prediction to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
