import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval_chartqa import (  # noqa: E402
    EvaluationConfig,
    evaluate_records,
    load_jsonl,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ChartQA prediction JSONL files.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--errors-output", type=Path, default=None)
    parser.add_argument("--evaluated-output", type=Path, default=None)
    parser.add_argument("--numeric-rel-tol", type=float, default=0.05)
    parser.add_argument("--numeric-abs-tol", type=float, default=1e-6)
    parser.add_argument("--allow-percent-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without reading or writing prediction files.",
    )
    return parser.parse_args()


def default_output_path(predictions: Path, suffix: str) -> Path:
    return predictions.with_name(f"{predictions.stem}_{suffix}{predictions.suffix}")


def main() -> int:
    args = parse_args()
    metrics_output = args.metrics_output or args.predictions.with_name(f"{args.predictions.stem}_metrics.json")
    errors_output = args.errors_output or default_output_path(args.predictions, "errors")
    evaluated_output = args.evaluated_output or default_output_path(args.predictions, "evaluated")

    if args.dry_run:
        print("Dry run OK.")
        print("predictions:", args.predictions)
        print("metrics_output:", metrics_output)
        print("errors_output:", errors_output)
        print("evaluated_output:", evaluated_output)
        print("numeric_rel_tol:", args.numeric_rel_tol)
        print("allow_percent_scale:", args.allow_percent_scale)
        return 0

    config = EvaluationConfig(
        numeric_rel_tol=args.numeric_rel_tol,
        numeric_abs_tol=args.numeric_abs_tol,
        allow_percent_scale=args.allow_percent_scale,
    )
    records = load_jsonl(args.predictions)
    metrics, evaluated, errors = evaluate_records(records, config)

    write_json(metrics_output, metrics)
    write_jsonl(errors_output, errors)
    write_jsonl(evaluated_output, evaluated)

    print("Predictions:", args.predictions)
    print("Total:", metrics["total"])
    print(f"Exact accuracy: {metrics['exact_match']}/{metrics['total']} = {metrics['exact_accuracy']:.2%}")
    print(
        "Relaxed accuracy: "
        f"{metrics['relaxed_correct']}/{metrics['total']} = {metrics['relaxed_accuracy']:.2%}"
    )
    print(
        "Relaxed numeric on numeric refs: "
        f"{metrics['relaxed_numeric_match']}/{metrics['numeric_reference_total']} = "
        f"{metrics['relaxed_numeric_accuracy_on_numeric']:.2%}"
    )
    print("Metrics output:", metrics_output)
    print("Errors output:", errors_output)
    print("Evaluated output:", evaluated_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

