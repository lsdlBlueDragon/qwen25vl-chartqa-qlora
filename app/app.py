import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import (  # noqa: E402
    DEFAULT_MODEL_ID,
    GenerationConfig,
    InferenceConfig,
    load_model_and_processor,
    predict,
)


BASE_MODE = "Base model"
ADAPTER_MODE = "Hardmix QLoRA"
DEFAULT_ADAPTER = "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
EXAMPLE_IMAGE = Path(__file__).resolve().parent / "examples" / "quarterly_sales.svg"
EXAMPLES = [
    [str(EXAMPLE_IMAGE), "Which region had the highest sales in Q4?", ADAPTER_MODE],
    [str(EXAMPLE_IMAGE), "What were North sales in Q3?", ADAPTER_MODE],
    [str(EXAMPLE_IMAGE), "What is the difference between North and West in Q2?", ADAPTER_MODE],
]


class ModelService:
    """Keep only one 3B model configuration active at a time."""

    def __init__(self) -> None:
        self.active_key: tuple[str, str | None] | None = None
        self.model: Any = None
        self.processor: Any = None

    def _clear(self) -> None:
        self.model = None
        self.processor = None
        self.active_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get(self, config: InferenceConfig) -> tuple[Any, Any]:
        key = (config.model_id, config.adapter_path)
        if self.active_key != key:
            self._clear()
            self.model, self.processor = load_model_and_processor(config)
            self.active_key = key
        return self.model, self.processor


MODEL_SERVICE = ModelService()


def adapter_path_from_env() -> str:
    return os.environ.get("CHARTQA_ADAPTER_PATH", DEFAULT_ADAPTER)


def inference_config_for_mode(mode: str) -> InferenceConfig:
    if mode == BASE_MODE:
        adapter_path = None
    elif mode == ADAPTER_MODE:
        adapter_path = adapter_path_from_env()
    else:
        raise ValueError(f"Unsupported model mode: {mode}")

    return InferenceConfig(
        model_id=os.environ.get("CHARTQA_MODEL_ID", DEFAULT_MODEL_ID),
        adapter_path=adapter_path,
        load_in_4bit=os.environ.get("CHARTQA_LOAD_IN_4BIT", "1") != "0",
        min_pixels=int(os.environ.get("CHARTQA_MIN_PIXELS", "50176")),
        max_pixels=int(os.environ.get("CHARTQA_MAX_PIXELS", "401408")),
    )


def answer_chart(image: Image.Image | None, question: str, mode: str) -> tuple[str, str]:
    if image is None:
        raise ValueError("Upload a chart image before running inference.")
    question = question.strip()
    if not question:
        raise ValueError("Enter a chart question before running inference.")

    config = inference_config_for_mode(mode)
    model, processor = MODEL_SERVICE.get(config)
    result = predict(
        image=image.convert("RGB"),
        question=question,
        model=model,
        processor=processor,
        inference_config=config,
        generation_config=GenerationConfig(max_new_tokens=64),
    )
    return result.answer, f"{result.latency_seconds:.2f} seconds"


def build_demo():
    import gradio as gr

    with gr.Blocks(title="Qwen2.5-VL ChartQA") as demo:
        gr.Markdown(
            "# Qwen2.5-VL ChartQA\n"
            "Ask a concise question about an uploaded chart. The hardmix adapter is the "
            "best relaxed-accuracy run on the frozen ChartQA validation benchmark."
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Chart image")
            with gr.Column():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="What is the maximum value?",
                )
                mode_input = gr.Radio(
                    choices=[ADAPTER_MODE, BASE_MODE],
                    value=ADAPTER_MODE,
                    label="Model",
                )
                submit = gr.Button("Answer", variant="primary")
        answer_output = gr.Textbox(label="Answer")
        latency_output = gr.Textbox(label="Generation latency")
        submit.click(
            fn=answer_chart,
            inputs=[image_input, question_input, mode_input],
            outputs=[answer_output, latency_output],
        )
        gr.Examples(
            examples=EXAMPLES,
            inputs=[image_input, question_input, mode_input],
            label="Built-in examples (self-created chart)",
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Qwen2.5-VL ChartQA Gradio demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build the UI without starting a server.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    demo = build_demo()
    if args.dry_run:
        print("Gradio app dry run OK.")
        print("Model ID:", os.environ.get("CHARTQA_MODEL_ID", DEFAULT_MODEL_ID))
        print("Adapter:", adapter_path_from_env())
        return 0
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
