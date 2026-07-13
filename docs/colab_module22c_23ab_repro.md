# Module 22C/23A/23B - Colab reproduction for local audit modules

This module fills the reproducibility gap for the local-only audit steps that were previously represented mostly as text cells in the notebook.

It does not load the model, does not use GPU, does not train, and does not run full-val. It restores existing Module 21/22B/23A/23B artifacts from Drive, runs the local audit scripts, and writes outputs back to Drive.

Recommended order after reconnect:

```text
22C23.0 -> 22C23.1 -> 22C23.2 -> 22C23.3 -> 22C23.4
```

## 22C23.0 Restore scripts and artifact inputs

```python
# Module 22C23.0 - restore scripts and existing artifacts for local audit reproduction
from pathlib import Path
import shutil

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print("Drive mount skipped or unavailable:", exc)

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
%cd /content/qwen25vl-chartqa-qlora

SCRIPT_NAMES = [
    "run_chartqa_staged_extraction_quality_audit.py",
    "run_chartqa_23a_cleanup_normalization_ablation.py",
    "run_chartqa_23b_hard_failure_diagnostics.py",
]
SCRIPT_DRIVE_DIRS = [
    DRIVE_ROOT / "scripts_module22c",
    DRIVE_ROOT / "scripts_module23a",
    DRIVE_ROOT / "scripts_module23b",
    DRIVE_ROOT / "scripts_module23c",
]

for script_name in SCRIPT_NAMES:
    local_script = REPO / "scripts" / script_name
    if local_script.exists():
        print("Script already present:", local_script)
        continue
    for drive_dir in SCRIPT_DRIVE_DIRS:
        candidate = drive_dir / script_name
        if candidate.exists():
            local_script.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, local_script)
            print("Restored script:", candidate, "->", local_script)
            break
    else:
        raise FileNotFoundError(f"Missing script in repo and Drive: {script_name}")

def copy_if_missing(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("Already present:", dst)
        return
    if not src.exists():
        raise FileNotFoundError(f"Missing required Drive file: {src}")
    shutil.copy2(src, dst)
    print("Restored:", src, "->", dst)

copy_if_missing(
    DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/data/chartqa_all_wrong_diagnostic_subset_85.jsonl",
    REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl",
)
copy_if_missing(
    DRIVE_ROOT / "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv",
    REPO / "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv",
)

# Restore 22B staged extraction files.
LOCAL_STAGED = REPO / "outputs/chartqa_all_wrong_diagnostics_from_drive/staged_extraction"
DRIVE_STAGED = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/staged_extraction"
LOCAL_STAGED.mkdir(parents=True, exist_ok=True)
if not DRIVE_STAGED.exists():
    raise FileNotFoundError(f"Missing Drive staged extraction directory: {DRIVE_STAGED}")
for src in DRIVE_STAGED.glob("*"):
    if src.is_file():
        shutil.copy2(src, LOCAL_STAGED / src.name)

# Restore Module 21 evaluated JSONL files into the standard local mirror.
LOCAL_EVAL = REPO / "outputs/chartqa_all_wrong_diagnostics_from_drive/evaluated"
LOCAL_EVAL.mkdir(parents=True, exist_ok=True)
needed = {
    "baseline_maxpix_802816_evaluated.jsonl",
    "f_maxpix_802816_evaluated.jsonl",
    "hardmix_axis_legend_prompt_802816_evaluated.jsonl",
    "hardmix_maxpix_602112_evaluated.jsonl",
    "hardmix_maxpix_802816_evaluated.jsonl",
    "image_plus_table_json_evaluated.jsonl",
    "table_json_only_evaluated.jsonl",
}
found = {}
for candidate in (DRIVE_ROOT / "outputs").rglob("*_evaluated.jsonl"):
    if candidate.name in needed and candidate.name not in found:
        found[candidate.name] = candidate
for name in sorted(needed):
    if name not in found:
        raise FileNotFoundError(f"Could not find evaluated JSONL in Drive outputs: {name}")
    shutil.copy2(found[name], LOCAL_EVAL / name)
    print("Restored evaluated:", found[name], "->", LOCAL_EVAL / name)

print("22C/23A/23B reproduction restore complete.")
```

## 22C23.1 Re-run 22C quality audit

```python
# Module 22C23.1 - reproduce 22C staged extraction quality audit
import subprocess

cmd = [
    "python", "scripts/run_chartqa_staged_extraction_quality_audit.py",
    "--staged-dir", "outputs/chartqa_all_wrong_diagnostics_from_drive/staged_extraction",
    "--module21-eval-dir", "outputs/chartqa_all_wrong_diagnostics_from_drive/evaluated",
    "--subset-jsonl", "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl",
    "--exclude-csv", "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv",
    "--output-dir", "outputs/chartqa_staged_extraction_quality_audit_22c",
    "--report-md", "docs/experiments/chartqa_staged_extraction_quality_audit_22c_colab_repro.md",
]
subprocess.run(cmd, check=True)
```

## 22C23.2 Re-run 23A cleanup + normalization-only ablation

```python
# Module 22C23.2 - reproduce 23A cleanup + normalization-only ablation
import subprocess

cmd = [
    "python", "scripts/run_chartqa_23a_cleanup_normalization_ablation.py",
    "--subset-jsonl", "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl",
    "--exclude-22a-csv", "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv",
    "--module21-eval-dir", "outputs/chartqa_all_wrong_diagnostics_from_drive/evaluated",
    "--module22b-eval-dir", "outputs/chartqa_all_wrong_diagnostics_from_drive/staged_extraction",
    "--output-dir", "outputs/chartqa_23a_cleanup_normalization",
    "--report-md", "docs/experiments/chartqa_23a_cleanup_normalization_colab_repro.md",
]
subprocess.run(cmd, check=True)
```

## 22C23.3 Re-run 23B hard-failure queue and restore Codex targeted labels

```python
# Module 22C23.3 - reproduce 23B hard-failure queue and restore Codex targeted review labels
from pathlib import Path
import shutil
import subprocess

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

cmd = [
    "python", "scripts/run_chartqa_23b_hard_failure_diagnostics.py",
    "--subset-jsonl", "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl",
    "--per-prediction-csv", "outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv",
    "--output-dir", "outputs/chartqa_23b_hard_failure_diagnostics",
    "--report-md", "docs/experiments/chartqa_23b_hard_failure_diagnostics_colab_repro.md",
]
subprocess.run(cmd, check=True)

# The Codex targeted review is a manual/visual audit artifact. Restore it for downstream 23C/24A.
src = DRIVE_ROOT / "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv"
dst = REPO / "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv"
if not src.exists():
    raise FileNotFoundError(f"Missing Codex targeted review CSV on Drive: {src}")
shutil.copy2(src, dst)
print("Restored Codex targeted review:", dst)
```

## 22C23.4 Sync reproduced audit outputs to Drive and read summaries

```python
# Module 22C23.4 - sync reproduced local-audit outputs and read summaries
from pathlib import Path
import json
import shutil

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

for rel in [
    "outputs/chartqa_staged_extraction_quality_audit_22c",
    "outputs/chartqa_23a_cleanup_normalization",
    "outputs/chartqa_23b_hard_failure_diagnostics",
]:
    local_dir = REPO / rel
    drive_dir = DRIVE_ROOT / rel
    drive_dir.mkdir(parents=True, exist_ok=True)
    for src in local_dir.rglob("*"):
        if src.is_file():
            dst = drive_dir / src.relative_to(local_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print("Synced:", local_dir, "->", drive_dir)

for path in [
    REPO / "outputs/chartqa_staged_extraction_quality_audit_22c/chartqa_22c_quality_audit_summary.json",
    REPO / "outputs/chartqa_23a_cleanup_normalization/chartqa_23a_summary.json",
    REPO / "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_summary.json",
]:
    print("\n==", path, "==")
    if path.exists():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)[:4000])
    else:
        print("missing")
```
