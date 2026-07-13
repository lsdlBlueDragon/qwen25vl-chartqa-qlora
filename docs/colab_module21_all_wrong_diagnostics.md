# Module 21 - ChartQA all-wrong diagnostic subset ablation

This module is designed for the active 3B Colab notebook. It does **not** run full validation. It only runs the recommended all-wrong diagnostic subset from the manual audit.

## Dependencies and minimal run order

Run after a fresh Colab restart:

```text
1.1 -> 1.3 -> 1.4 -> 21.1 -> 21.2 -> 21.3 -> 21.4 -> 21.5 -> 21.6
```

Assumptions:

- Google Drive is mounted at `/content/drive/MyDrive`.
- Repo path is `/content/qwen25vl-chartqa-qlora`.
- Drive project root is `/content/drive/MyDrive/qwen25vl-chartqa-qlora`.
- Module 21 helper scripts are either already in `scripts/` after `git pull`, or backed up in Drive under `scripts_module21/`.
- Manual audit inputs are backed up in Drive under `outputs/chartqa_all_wrong_diagnostics/inputs/`.
- This module writes local outputs under `outputs/chartqa_all_wrong_diagnostics/` and persists all important outputs to Drive.

Long-running cells use progress bars and are safely rerunnable. Prediction/extraction JSONL files are appended incrementally and skip completed `sample_index` rows on rerun.

## 21.1 Restore Module 21 inputs, scripts, and adapters

```python
# Module 21.1 - restore inputs, helper scripts, and required adapters
from pathlib import Path
import json
import shutil
import subprocess

from tqdm.auto import tqdm

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print("Drive mount skipped or unavailable:", exc)

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_INPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/inputs"
DRIVE_SCRIPT_DIR = DRIVE_ROOT / "scripts_module21"
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics"

LOCAL_OUTPUT_DIR = REPO / "outputs/chartqa_all_wrong_diagnostics"
LOCAL_DATA_DIR = REPO / "data/diagnostics"
LOCAL_SCRIPT_DIR = REPO / "scripts"

MODULE21_SCRIPTS = [
    "prepare_chartqa_all_wrong_subset.py",
    "run_chartqa_subset_ablation.py",
    "run_chartqa_structured_extraction_diagnostic.py",
    "summarize_chartqa_all_wrong_diagnostics.py",
]

REQUIRED_INPUTS = [
    "chartqa_all_wrong_manual_audit_report.md",
    "chartqa_all_wrong_manual_audit_table.csv",
    "chartqa_all_wrong_manual_audit_table.json",
    "chartqa_all_wrong_recommended_diagnostic_subset.csv",
]

FULL_VAL_SFT = DRIVE_ROOT / "data/processed/chartqa_val_full_sft_1920.jsonl"
FULL_VAL_IMAGE_ROOT = DRIVE_ROOT / "data/processed/chartqa_val_full_sft_1920_images"

HARDMIX_ADAPTER = REPO / "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
F_ADAPTER = REPO / "outputs/adapters/chartqa_qlora_train1k_steps250_r16a32"
DRIVE_HARDMIX_ADAPTER = DRIVE_ROOT / "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
DRIVE_F_ADAPTER = DRIVE_ROOT / "outputs/adapters/chartqa_qlora_train1k_steps250_r16a32"

for path in [REPO, DRIVE_ROOT, DRIVE_INPUT_DIR, FULL_VAL_SFT, FULL_VAL_IMAGE_ROOT]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

%cd /content/qwen25vl-chartqa-qlora

# If this notebook is running before the new scripts are in GitHub, restore them from Drive.
LOCAL_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
for script_name in MODULE21_SCRIPTS:
    local_script = LOCAL_SCRIPT_DIR / script_name
    drive_script = DRIVE_SCRIPT_DIR / script_name
    if local_script.exists():
        print(f"Script already present: {local_script}")
    elif drive_script.exists():
        shutil.copy2(drive_script, local_script)
        print(f"Restored script from Drive: {drive_script} -> {local_script}")
    else:
        raise FileNotFoundError(f"Missing Module 21 script in repo and Drive: {script_name}")

for file_name in REQUIRED_INPUTS:
    path = DRIVE_INPUT_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing manual audit input: {path}")
    print("Input OK:", path)

def copytree_with_progress(src: Path, dst: Path, desc: str) -> None:
    if dst.exists():
        print(f"Adapter already restored: {dst}")
        return
    files = [p for p in src.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in adapter source: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for file_path in tqdm(files, desc=desc, unit="files"):
        rel = file_path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target)

copytree_with_progress(DRIVE_HARDMIX_ADAPTER, HARDMIX_ADAPTER, "Restoring hardmix adapter")
copytree_with_progress(DRIVE_F_ADAPTER, F_ADAPTER, "Restoring F adapter")

LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Module 21 restore complete.")
print("Drive output dir:", DRIVE_OUTPUT_DIR)
print("Hardmix adapter:", HARDMIX_ADAPTER)
print("F adapter:", F_ADAPTER)
```

## 21.2 Prepare diagnostic subset

```python
# Module 21.2 - prepare the fixed all-wrong diagnostic subset
from pathlib import Path
import subprocess

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_INPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/inputs"
DRIVE_DATA_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/data"

SUBSET_JSONL = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"
SUBSET_SUMMARY = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85_summary.json"

cmd = [
    "python", "scripts/prepare_chartqa_all_wrong_subset.py",
    "--subset-csv", str(DRIVE_INPUT_DIR / "chartqa_all_wrong_recommended_diagnostic_subset.csv"),
    "--manual-audit-csv", str(DRIVE_INPUT_DIR / "chartqa_all_wrong_manual_audit_table.csv"),
    "--full-val-sft-jsonl", str(DRIVE_ROOT / "data/processed/chartqa_val_full_sft_1920.jsonl"),
    "--image-root", str(DRIVE_ROOT / "data/processed/chartqa_val_full_sft_1920_images"),
    "--output-jsonl", str(SUBSET_JSONL),
    "--summary-output", str(SUBSET_SUMMARY),
    "--drive-output-dir", str(DRIVE_DATA_DIR),
]
subprocess.run(cmd, check=True)
```

## 21.3 Snapshot original all-wrong predictions for the subset

```python
# Module 21.3 - summarize existing original predictions from the manual audit
from pathlib import Path
import csv
import json
import shutil
from collections import Counter

from tqdm.auto import tqdm

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_INPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/inputs"
DRIVE_SUMMARY_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/summaries"
LOCAL_SUMMARY_DIR = REPO / "outputs/chartqa_all_wrong_diagnostics/summaries"
LOCAL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

manual_csv = DRIVE_INPUT_DIR / "chartqa_all_wrong_recommended_diagnostic_subset.csv"
rows = []
with manual_csv.open("r", encoding="utf-8-sig", newline="") as handle:
    for row in tqdm(csv.DictReader(handle), desc="Reading diagnostic subset rows", unit="rows"):
        rows.append(row)

summary = {
    "total": len(rows),
    "reviewed_primary_counts": dict(Counter(row["reviewed_primary"] for row in rows)),
    "review_flag_counts": dict(Counter(flag for row in rows for flag in row["review_flags"].split(";") if flag)),
    "note": "These rows are the fixed all-runs-wrong diagnostic subset. Original seven full-val runs are all relaxed-wrong by construction.",
}

json_path = LOCAL_SUMMARY_DIR / "original_all_wrong_subset_snapshot.json"
json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
shutil.copy2(json_path, DRIVE_SUMMARY_DIR / json_path.name)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Copied snapshot to:", DRIVE_SUMMARY_DIR / json_path.name)
```

## 21.4 High-resolution and prompt ablation

```python
# Module 21.4 - run high-resolution/prompt ablation on the fixed subset
from pathlib import Path
import subprocess

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

SUBSET_JSONL = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"
HARDMIX_ADAPTER = REPO / "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
F_ADAPTER = REPO / "outputs/adapters/chartqa_qlora_train1k_steps250_r16a32"

cmd = [
    "python", "scripts/run_chartqa_subset_ablation.py",
    "--subset-jsonl", str(SUBSET_JSONL),
    "--output-dir", "outputs/chartqa_all_wrong_diagnostics",
    "--drive-output-dir", str(DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/ablation_runs"),
    "--hardmix-adapter-path", str(HARDMIX_ADAPTER),
    "--f-adapter-path", str(F_ADAPTER),
    "--runs",
    "baseline_maxpix_802816",
    "hardmix_maxpix_602112",
    "hardmix_maxpix_802816",
    "f_maxpix_802816",
    "hardmix_axis_legend_prompt_802816",
]
subprocess.run(cmd, check=True)
```

## 21.5 Structured extraction diagnostic

```python
# Module 21.5 - chart-to-JSON extraction plus table-assisted QA
from pathlib import Path
import subprocess

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
SUBSET_JSONL = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"

# Use the base 3B model first. The QA adapters were not trained for chart-to-JSON extraction.
cmd = [
    "python", "scripts/run_chartqa_structured_extraction_diagnostic.py",
    "--subset-jsonl", str(SUBSET_JSONL),
    "--output-dir", "outputs/chartqa_all_wrong_diagnostics",
    "--drive-output-dir", str(DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/structured_extraction"),
    "--max-pixels", "802816",
    "--extract-max-new-tokens", "768",
    "--qa-max-new-tokens", "128",
]
subprocess.run(cmd, check=True)
```

## 21.6 Summarize diagnostic results

```python
# Module 21.6 - summarize recovered samples across diagnostic runs
from pathlib import Path
import subprocess

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
SUBSET_JSONL = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"

cmd = [
    "python", "scripts/summarize_chartqa_all_wrong_diagnostics.py",
    "--subset-jsonl", str(SUBSET_JSONL),
    "--output-dir", "outputs/chartqa_all_wrong_diagnostics",
    "--drive-output-dir", str(DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/summaries"),
]
subprocess.run(cmd, check=True)
```

