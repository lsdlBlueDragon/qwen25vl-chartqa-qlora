import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationConfig:
    numeric_rel_tol: float = 0.05
    numeric_abs_tol: float = 1e-6
    allow_percent_scale: bool = True


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def first_reference(record: dict[str, Any]) -> Any:
    if "reference_answer" in record:
        return record["reference_answer"]
    labels = record.get("all_labels") or record.get("label")
    if isinstance(labels, list) and labels:
        return labels[0]
    return labels


def prediction_text(record: dict[str, Any]) -> Any:
    if "answer" in record:
        return record["answer"]
    return record.get("prediction")


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"[,，]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match(prediction: Any, reference: Any) -> bool:
    return normalize_text(prediction) == normalize_text(reference)


def parse_number(value: Any) -> float | None:
    text = normalize_text(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def numbers_close(prediction: float, reference: float, config: EvaluationConfig) -> bool:
    if math.isclose(prediction, reference, rel_tol=config.numeric_rel_tol, abs_tol=config.numeric_abs_tol):
        return True

    if not config.allow_percent_scale:
        return False

    # ChartQA 中会出现 0.72 和 72 这种百分比尺度差异。这里仅在 relaxed numeric 中放宽，
    # exact match 仍保持严格，便于区分格式问题和真正读图错误。
    scaled_candidates = [
        (prediction / 100.0, reference),
        (prediction, reference / 100.0),
        (prediction * 100.0, reference),
        (prediction, reference * 100.0),
    ]
    return any(
        math.isclose(left, right, rel_tol=config.numeric_rel_tol, abs_tol=config.numeric_abs_tol)
        for left, right in scaled_candidates
    )


def relaxed_numeric_match(prediction: Any, reference: Any, config: EvaluationConfig) -> bool:
    pred_number = parse_number(prediction)
    ref_number = parse_number(reference)
    if pred_number is None or ref_number is None:
        return False
    return numbers_close(pred_number, ref_number, config)


def classify_record(record: dict[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    prediction = prediction_text(record)
    reference = first_reference(record)
    is_exact = exact_match(prediction, reference)
    is_numeric = parse_number(reference) is not None
    is_relaxed_numeric = relaxed_numeric_match(prediction, reference, config) if is_numeric else False
    is_relaxed_correct = is_exact or is_relaxed_numeric

    return {
        **record,
        "eval_prediction": prediction,
        "eval_reference": reference,
        "normalized_prediction": normalize_text(prediction),
        "normalized_reference": normalize_text(reference),
        "eval_exact_match": is_exact,
        "eval_reference_is_numeric": is_numeric,
        "eval_relaxed_numeric_match": is_relaxed_numeric,
        "eval_relaxed_correct": is_relaxed_correct,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    exact = sum(1 for item in records if item["eval_exact_match"])
    relaxed_numeric = sum(1 for item in records if item["eval_relaxed_numeric_match"])
    relaxed_correct = sum(1 for item in records if item["eval_relaxed_correct"])
    numeric_total = sum(1 for item in records if item["eval_reference_is_numeric"])
    latencies = [
        float(item["latency_seconds"])
        for item in records
        if item.get("latency_seconds") is not None
    ]

    by_human_or_machine: dict[str, dict[str, Any]] = {}
    for item in records:
        key = str(item.get("human_or_machine", "unknown"))
        group = by_human_or_machine.setdefault(key, {"total": 0, "exact": 0, "relaxed_correct": 0})
        group["total"] += 1
        group["exact"] += int(item["eval_exact_match"])
        group["relaxed_correct"] += int(item["eval_relaxed_correct"])

    for group in by_human_or_machine.values():
        group["exact_accuracy"] = group["exact"] / group["total"] if group["total"] else 0.0
        group["relaxed_accuracy"] = (
            group["relaxed_correct"] / group["total"] if group["total"] else 0.0
        )

    return {
        "total": total,
        "exact_match": exact,
        "exact_accuracy": exact / total if total else 0.0,
        "numeric_reference_total": numeric_total,
        "relaxed_numeric_match": relaxed_numeric,
        "relaxed_numeric_accuracy_on_all": relaxed_numeric / total if total else 0.0,
        "relaxed_numeric_accuracy_on_numeric": relaxed_numeric / numeric_total if numeric_total else 0.0,
        "relaxed_correct": relaxed_correct,
        "relaxed_accuracy": relaxed_correct / total if total else 0.0,
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "by_human_or_machine": by_human_or_machine,
    }


def evaluate_records(
    records: list[dict[str, Any]],
    config: EvaluationConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or EvaluationConfig()
    evaluated = [classify_record(record, config) for record in records]
    metrics = summarize(evaluated)
    errors = [record for record in evaluated if not record["eval_relaxed_correct"]]
    return metrics, evaluated, errors

