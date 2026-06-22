import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import DEFAULT_MODEL_ID  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small Qwen2.5-VL QLoRA smoke training job.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/chartqa_train_sft_100.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapters/chartqa_qlora_smoke"))
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=50_176)
    parser.add_argument("--max-pixels", type=int, default=401_408)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and arguments without loading the model or starting training.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_jsonl_images(path: Path, records: list[dict[str, Any]]) -> None:
    missing = []
    for record in records[:10]:
        image_path = path.parent / record["image"]
        if not image_path.exists():
            missing.append(str(image_path))
    if missing:
        raise FileNotFoundError(f"Missing image files referenced by {path}: {missing[:3]}")


class ChartQASFTDataset:
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.base_dir = jsonl_path.parent
        self.records = load_jsonl(jsonl_path)
        validate_jsonl_images(jsonl_path, self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def messages_with_loaded_image(record: dict[str, Any], include_assistant: bool):
    from PIL import Image

    image_path = record["_base_dir"] / record["image"]
    image = Image.open(image_path).convert("RGB")
    messages = record["messages"] if include_assistant else record["messages"][:1]

    loaded_messages = []
    for message in messages:
        content = []
        for item in message["content"]:
            if item.get("type") == "image":
                content.append({"type": "image", "image": image})
            else:
                content.append(item)
        loaded_messages.append({"role": message["role"], "content": content})
    return loaded_messages


@dataclass
class QwenVLSFTCollator:
    processor: Any

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from qwen_vl_utils import process_vision_info

        full_messages_batch = []
        prompt_messages_batch = []
        full_texts = []
        prompt_texts = []

        for record in records:
            full_messages = messages_with_loaded_image(record, include_assistant=True)
            prompt_messages = messages_with_loaded_image(record, include_assistant=False)
            full_messages_batch.append(full_messages)
            prompt_messages_batch.append(prompt_messages)
            full_texts.append(
                self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            )
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            )

        full_image_inputs, _ = process_vision_info(full_messages_batch)
        prompt_image_inputs, _ = process_vision_info(prompt_messages_batch)

        inputs = self.processor(
            text=full_texts,
            images=full_image_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=prompt_image_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        labels[labels == pad_token_id] = -100

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100

        inputs["labels"] = labels
        return inputs


def attach_base_dir(dataset: ChartQASFTDataset) -> None:
    for record in dataset.records:
        record["_base_dir"] = dataset.base_dir


def copy_adapter(output_dir: Path, drive_output_dir: Path) -> Path:
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    drive_adapter_dir = drive_output_dir / output_dir.name
    shutil.copytree(output_dir, drive_adapter_dir, dirs_exist_ok=True)
    return drive_adapter_dir


def main() -> int:
    args = parse_args()

    if args.dry_run:
        train_records = load_jsonl(args.train_jsonl) if args.train_jsonl.exists() else []
        eval_records = load_jsonl(args.eval_jsonl) if args.eval_jsonl and args.eval_jsonl.exists() else []
        if train_records:
            validate_jsonl_images(args.train_jsonl, train_records)
        if args.eval_jsonl and eval_records:
            validate_jsonl_images(args.eval_jsonl, eval_records)
        print("Dry run OK.")
        print("train_jsonl:", args.train_jsonl)
        print("train_records:", len(train_records))
        print("eval_jsonl:", args.eval_jsonl)
        print("eval_records:", len(eval_records))
        print("output_dir:", args.output_dir)
        print("drive_output_dir:", args.drive_output_dir)
        print("model_id:", args.model_id)
        print("max_steps:", args.max_steps)
        print("load_in_4bit:", args.load_in_4bit)
        return 0

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        Qwen2_5_VLForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Switch Colab runtime to GPU before training.")

    train_dataset = ChartQASFTDataset(args.train_jsonl)
    attach_base_dir(train_dataset)
    eval_dataset = None
    if args.eval_jsonl:
        eval_dataset = ChartQASFTDataset(args.eval_jsonl)
        attach_base_dir(eval_dataset)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("Loading model:", args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_strategy="steps" if args.save_steps > 0 else "no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=QwenVLSFTCollator(processor),
    )

    print("Starting QLoRA smoke training...")
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Saved adapter and processor to:", args.output_dir)

    if args.drive_output_dir:
        drive_adapter_dir = copy_adapter(args.output_dir, args.drive_output_dir)
        print("Copied adapter to Drive:", drive_adapter_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
