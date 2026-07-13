import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_prepare_chartqa_sft_dry_run_does_not_write_output(tmp_path):
    output_path = tmp_path / "prepared.jsonl"

    result = run_cli(
        "scripts/prepare_chartqa_sft.py",
        "--split",
        "val",
        "--n-samples",
        "3",
        "--output",
        str(output_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run OK." in result.stdout
    assert "split: val" in result.stdout
    assert "n_samples: 3" in result.stdout
    assert not output_path.exists()
    assert not output_path.with_name("prepared_images").exists()


def test_single_image_dry_run_does_not_require_image_or_write_output(tmp_path):
    missing_image = tmp_path / "not_downloaded.png"
    output_path = tmp_path / "prediction.jsonl"

    result = run_cli(
        "scripts/run_baseline_image.py",
        "--image",
        str(missing_image),
        "--question",
        "What is the maximum value?",
        "--output",
        str(output_path),
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "Dry run OK." in result.stdout
    assert "What is the maximum value?" in result.stdout
    assert not output_path.exists()
