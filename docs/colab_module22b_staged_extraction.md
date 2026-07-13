# Module 22B - staged chart-to-table extraction diagnostic

This module runs staged chart-to-table extraction on the same all-wrong diagnostic subset. It does **not** run full validation and does **not** train a new adapter.

Compared with Module 21.5, this module avoids one-shot chart-to-JSON. It splits the extraction into:

1. chart overview;
2. axes / ticks / legend / color mapping;
3. question-relevant data table;
4. table-only QA;
5. image + staged table QA.

By default it skips the 8 high-priority `exclude_or_fix_reference` samples produced by Module 22A, leaving 77 valid samples for model-capability diagnostics.

## Dependencies and minimal run order

Run after a fresh Colab restart:

```text
1.1 -> 1.3 -> 1.4 -> 22B.1 -> 22B.2 -> 22B.3
```

Required Drive inputs:

- `/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_all_wrong_diagnostics/data/chartqa_all_wrong_diagnostic_subset_85.jsonl`
- `/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv`
- helper script in repo `scripts/` or Drive `scripts_module22b/`.

Long-running steps use progress bars. Stage outputs are append-only JSONL files and are written to both local Colab and Drive, so reruns skip completed `sample_index` rows.

## 22B.1 Restore inputs and helper script

```python
# Module 22B.1 - 恢复 staged extraction 输入和 helper 脚本
from pathlib import Path
import shutil

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print("Drive mount skipped or unavailable:", exc)

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_SCRIPT_DIR = DRIVE_ROOT / "scripts_module22b"
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/staged_extraction"

DRIVE_SUBSET_JSONL = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/data/chartqa_all_wrong_diagnostic_subset_85.jsonl"
DRIVE_EXCLUDE_CSV = DRIVE_ROOT / "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv"

LOCAL_SUBSET_JSONL = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"
LOCAL_EXCLUDE_CSV = REPO / "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv"

SCRIPT_NAME = "run_chartqa_staged_extraction_diagnostic.py"
LOCAL_SCRIPT = REPO / "scripts" / SCRIPT_NAME
DRIVE_SCRIPT = DRIVE_SCRIPT_DIR / SCRIPT_NAME

for path in [REPO, DRIVE_ROOT, DRIVE_SUBSET_JSONL, DRIVE_EXCLUDE_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

%cd /content/qwen25vl-chartqa-qlora

LOCAL_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
if LOCAL_SCRIPT.exists():
    print("Script already present:", LOCAL_SCRIPT)
elif DRIVE_SCRIPT.exists():
    shutil.copy2(DRIVE_SCRIPT, LOCAL_SCRIPT)
    print("Restored script from Drive:", DRIVE_SCRIPT, "->", LOCAL_SCRIPT)
else:
    raise FileNotFoundError(f"Missing helper script in repo and Drive: {SCRIPT_NAME}")

LOCAL_SUBSET_JSONL.parent.mkdir(parents=True, exist_ok=True)
LOCAL_EXCLUDE_CSV.parent.mkdir(parents=True, exist_ok=True)
if not LOCAL_SUBSET_JSONL.exists():
    shutil.copy2(DRIVE_SUBSET_JSONL, LOCAL_SUBSET_JSONL)
    print("Restored subset JSONL:", LOCAL_SUBSET_JSONL)
if not LOCAL_EXCLUDE_CSV.exists():
    shutil.copy2(DRIVE_EXCLUDE_CSV, LOCAL_EXCLUDE_CSV)
    print("Restored exclude list:", LOCAL_EXCLUDE_CSV)

DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Module 22B restore complete.")
print("Subset JSONL:", LOCAL_SUBSET_JSONL)
print("Exclude list:", LOCAL_EXCLUDE_CSV)
print("Drive output dir:", DRIVE_OUTPUT_DIR)
```

## 22B.2 Run staged extraction and QA

```python
# Module 22B.2 - 分步 chart-to-table extraction + QA
from pathlib import Path
import subprocess

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

SUBSET_JSONL = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"
EXCLUDE_CSV = REPO / "outputs/chartqa_evaluator_cleanup/chartqa_subset85_exclude_or_fix_reference_list.csv"
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/staged_extraction"

cmd = [
    "python", "scripts/run_chartqa_staged_extraction_diagnostic.py",
    "--subset-jsonl", str(SUBSET_JSONL),
    "--exclude-list-csv", str(EXCLUDE_CSV),
    "--output-dir", "outputs/chartqa_all_wrong_diagnostics/staged_extraction",
    "--drive-output-dir", str(DRIVE_OUTPUT_DIR),
    "--max-pixels", "802816",
    "--stage-max-new-tokens", "768",
    "--qa-max-new-tokens", "128",
]
subprocess.run(cmd, check=True)
```

## 22B.3 Read staged extraction summary

```python
# Module 22B.3 - 读取 staged extraction 结果
from pathlib import Path
import json

REPO = Path("/content/qwen25vl-chartqa-qlora")
SUMMARY_JSON = REPO / "outputs/chartqa_all_wrong_diagnostics/staged_extraction/staged_extraction_summary.json"
REPORT_MD = REPO / "outputs/chartqa_all_wrong_diagnostics/staged_extraction/staged_extraction_report.md"

summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

print("有效样本数（剔除 reference/evaluator 高风险样本后）:", summary["subset_total_after_excluding_reference_issues"])
print("跳过样本:", summary["skipped_exclude_or_fix_reference_indices"])
print("stage JSON validity:")
print(json.dumps(summary["stage_valid_json"], ensure_ascii=False, indent=2))
print("QA runs:")
for run in summary["runs"]:
    print(
        run["run_name"],
        f"{run['relaxed_correct']}/{run['total']} = {run['relaxed_accuracy']:.2%}",
        "recovered:",
        run["recovered_indices"],
    )
print("oracle:", f"{summary['oracle_recovered_count']}/{summary['subset_total_after_excluding_reference_issues']} = {summary['oracle_recovered_accuracy']:.2%}")
print("still wrong:", summary["still_wrong_count"])
print("\nReport:", REPORT_MD)
print(REPORT_MD.read_text(encoding="utf-8")[:4000])
```

## 22B.4 中文说明

本模块验证的是“分步结构化读图”是否比 Module 21 的一次性 chart-to-JSON 更稳。

判断口径：

- 如果 `overview/axes_legend/data_table` 的 JSON 合法率明显高于 Module 21 的一次性 extraction，说明分步 schema 更可控。
- 如果 `staged_table_json_only` 追回更多样本，说明结构化抽取本身有效。
- 如果 `staged_image_plus_table_json` 更好，说明 extraction 仍不完整，图像二次校验有帮助。
- 如果二者都不如 Module 21 的 one-shot extraction，则应先人工审查 staged prompts，而不是扩到 full-val。

本模块仍然不建议训练新 LoRA，也不建议 full-val 扩展。只有在 77 条有效 subset 上看到清晰收益后，才进入更大规模验证。

