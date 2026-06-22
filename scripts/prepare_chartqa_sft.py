import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = (
    "Answer the chart question with a concise answer. "
    "If the answer is numeric, return only the number and unit when needed.\n"
    "Question: {question}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ChartQA samples for Qwen2.5-VL SFT.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--drive-output-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without loading ChartQA or writing files.",
    )
    return parser.parse_args()


def image_dir_for_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_images")


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    return Path(f"data/processed/chartqa_{args.split}_sft_{args.n_samples}.jsonl")


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def first_answer(labels: Any) -> str:
    if isinstance(labels, list):
        if not labels:
            return ""
        return str(labels[0])
    return str(labels)


def build_record(
    *,
    sample: dict[str, Any],
    sample_index: int,
    split: str,
    image_path: Path,
) -> dict[str, Any]:
    question = str(sample["query"])
    answer = first_answer(sample["label"])
    image_value = image_path.as_posix()

    return {
        "sample_index": sample_index,
        "split": split,
        "human_or_machine": sample.get("human_or_machine"),
        "query": question,
        "answer": answer,
        "all_labels": sample["label"],
        "image": image_value,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_value},
                    {"type": "text", "text": format_prompt(question)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_outputs(output_path: Path, image_dir: Path, drive_output_dir: Path) -> tuple[Path, Path]:
    drive_output_dir.mkdir(parents=True, exist_ok=True)
    drive_jsonl = drive_output_dir / output_path.name
    drive_image_dir = drive_output_dir / image_dir.name
    shutil.copy2(output_path, drive_jsonl)
    shutil.copytree(image_dir, drive_image_dir, dirs_exist_ok=True)
    return drive_jsonl, drive_image_dir


def main() -> int:
    args = parse_args()
    output_path = resolve_output_path(args)
    image_dir = image_dir_for_output(output_path)

    if args.dry_run:
        print("Dry run OK.")
        print("split:", args.split)
        print("n_samples:", args.n_samples)
        print("output:", output_path)
        print("image_dir:", image_dir)
        print("drive_output_dir:", args.drive_output_dir)
        return 0

    from datasets import load_dataset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ChartQA streaming dataset split={args.split!r}...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split=args.split, streaming=True)

    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(dataset):
        if idx >= args.n_samples:
            break

        image_name = f"{args.split}_{idx:06d}.png"
        image_path = image_dir / image_name
        sample["image"].convert("RGB").save(image_path)

        relative_image_path = Path(image_dir.name) / image_name
        record = build_record(
            sample=sample,
            sample_index=idx,
            split=args.split,
            image_path=relative_image_path,
        )
        records.append(record)

    write_jsonl(output_path, records)

    print(f"Wrote {len(records)} SFT records to {output_path}")
    print(f"Wrote images to {image_dir}")

    if args.drive_output_dir:
        drive_jsonl, drive_image_dir = copy_outputs(output_path, image_dir, args.drive_output_dir)
        print(f"Copied JSONL to Drive: {drive_jsonl}")
        print(f"Copied images to Drive: {drive_image_dir}")

    print("\nPreview:")
    for record in records[:3]:
        print(
            {
                "sample_index": record["sample_index"],
                "image": record["image"],
                "query": record["query"],
                "answer": record["answer"],
                "human_or_machine": record["human_or_machine"],
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
