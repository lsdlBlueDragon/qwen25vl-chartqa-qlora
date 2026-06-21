import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
    "qwen-vl-utils": "qwen_vl_utils",
    "gradio": "gradio",
    "Pillow": "PIL",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyyaml": "yaml",
}


def package_status(dist_name: str, import_name: str) -> dict:
    status = {
        "installed": False,
        "importable": False,
        "version": None,
        "error": None,
    }
    try:
        status["version"] = importlib.metadata.version(dist_name)
        status["installed"] = True
    except importlib.metadata.PackageNotFoundError:
        pass

    try:
        importlib.import_module(import_name)
        status["importable"] = True
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"

    return status


def torch_status() -> dict:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    cuda = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    if cuda["available"]:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )

    return {
        "available": True,
        "version": torch.__version__,
        "cuda": cuda,
    }


def build_report() -> dict:
    return {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "packages": {
            dist: package_status(dist, import_name)
            for dist, import_name in PACKAGES.items()
        },
        "torch_runtime": torch_status(),
        "project_policy": {
            "local_role": "non-training smoke tests only",
            "training_role": "Colab GPU",
            "deployment_role": "Hugging Face GPU Space",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local/Colab/Space runtime readiness.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when any required package is missing or not importable.",
    )
    args = parser.parse_args()

    report = build_report()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    if args.strict:
        failed = [
            name
            for name, item in report["packages"].items()
            if not item["installed"] or not item["importable"]
        ]
        if failed:
            print("Missing or non-importable packages:", ", ".join(failed), file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

