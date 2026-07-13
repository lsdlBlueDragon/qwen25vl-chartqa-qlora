from pathlib import Path

from scripts.prepare_chartqa_sft import build_record, format_prompt


def test_build_record_preserves_metadata_and_qwen_message_shape():
    sample = {
        "query": "What is the highest value?",
        "label": ["42", "42.0"],
        "human_or_machine": 0,
    }

    record = build_record(
        sample=sample,
        sample_index=7,
        split="train",
        image_path=Path("chartqa_train_images/train_000007.png"),
    )

    assert record["sample_index"] == 7
    assert record["split"] == "train"
    assert record["human_or_machine"] == 0
    assert record["answer"] == "42"
    assert record["all_labels"] == ["42", "42.0"]
    assert record["image"] == "chartqa_train_images/train_000007.png"
    assert record["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "chartqa_train_images/train_000007.png"},
                {"type": "text", "text": format_prompt(sample["query"])},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "42"}],
        },
    ]
