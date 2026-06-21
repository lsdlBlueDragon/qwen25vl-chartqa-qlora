import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


@dataclass
class InferenceConfig:
    model_id: str = DEFAULT_MODEL_ID
    adapter_path: str | None = None
    load_in_4bit: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"
    min_pixels: int = 50_176
    max_pixels: int = 401_408


@dataclass
class PredictionResult:
    question: str
    answer: str
    model_id: str
    adapter_path: str | None
    latency_seconds: float
    generation: dict[str, Any]
    image_path: str | None = None


def load_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def build_messages(image: Image.Image, question: str) -> list[dict[str, Any]]:
    # 所有入口共用同一份 prompt，保证 baseline、QLoRA adapter 和 Space demo 结果可比较。
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Answer the chart question with a concise answer. "
                        "If the answer is numeric, return only the number and unit when needed.\n"
                        f"Question: {question}"
                    ),
                },
            ],
        }
    ]


def _torch_dtype_from_name(dtype_name: str):
    import torch

    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_processor(config: InferenceConfig):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization_config = None
    if config.load_in_4bit:
        # Colab T4/L4 上优先用 4-bit NF4，先保证 3B VLM 能稳定加载和推理。
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        torch_dtype=_torch_dtype_from_name(config.torch_dtype),
        device_map=config.device_map,
        quantization_config=quantization_config,
    )

    if config.adapter_path:
        from peft import PeftModel

        # LoRA adapter 只挂在 base model 上，不改变 baseline 加载路径，方便做 A/B 对比。
        model = PeftModel.from_pretrained(model, config.adapter_path)

    processor = AutoProcessor.from_pretrained(
        config.model_id,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    return model, processor


def predict(
    image: Image.Image,
    question: str,
    model: Any,
    processor: Any,
    inference_config: InferenceConfig,
    generation_config: GenerationConfig | None = None,
    image_path: str | None = None,
) -> PredictionResult:
    import torch
    from qwen_vl_utils import process_vision_info

    generation_config = generation_config or GenerationConfig()
    messages = build_messages(image, question)
    # Qwen2.5-VL 的文本模板和视觉输入必须来自同一份 messages，否则 image token 会对不上。
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

    generate_kwargs = asdict(generation_config)
    start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)
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

    return PredictionResult(
        question=question,
        answer=answer,
        model_id=inference_config.model_id,
        adapter_path=inference_config.adapter_path,
        latency_seconds=round(latency, 4),
        generation=generate_kwargs,
        image_path=image_path,
    )


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
