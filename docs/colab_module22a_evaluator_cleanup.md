# Module 22A - evaluator/data cleanup list

This module does not run model inference and does not run full validation. It only turns the manual all-wrong audit into a reproducible evaluator/data cleanup list.

## Dependencies and minimal run order

Run after a fresh Colab restart:

```text
1.1 -> 1.3 -> 1.4 -> 22A.1 -> 22A.2 -> 22A.3
```

Inputs must already exist in Drive:

- `/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_all_wrong_diagnostics/inputs/chartqa_all_wrong_manual_audit_table.csv`
- `/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_all_wrong_diagnostics/inputs/chartqa_all_wrong_recommended_diagnostic_subset.csv`
- Module 22A helper script in either repo `scripts/` or Drive `scripts_module22a/`.

Outputs are written locally and copied to Drive:

```text
outputs/chartqa_evaluator_cleanup/
/content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_evaluator_cleanup/
```

## 22A.1 Restore inputs and helper script

```python
# Module 22A.1 - 恢复输入和 helper 脚本；不跑模型，不跑 full-val
from pathlib import Path
import shutil

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print("Drive mount skipped or unavailable:", exc)

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_INPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/inputs"
DRIVE_SCRIPT_DIR = DRIVE_ROOT / "scripts_module22a"
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_evaluator_cleanup"
LOCAL_OUTPUT_DIR = REPO / "outputs/chartqa_evaluator_cleanup"

SCRIPT_NAME = "prepare_chartqa_evaluator_cleanup.py"
LOCAL_SCRIPT = REPO / "scripts" / SCRIPT_NAME
DRIVE_SCRIPT = DRIVE_SCRIPT_DIR / SCRIPT_NAME

MANUAL_AUDIT_CSV = DRIVE_INPUT_DIR / "chartqa_all_wrong_manual_audit_table.csv"
SUBSET_CSV = DRIVE_INPUT_DIR / "chartqa_all_wrong_recommended_diagnostic_subset.csv"

for path in [REPO, DRIVE_ROOT, DRIVE_INPUT_DIR, MANUAL_AUDIT_CSV, SUBSET_CSV]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

%cd /content/qwen25vl-chartqa-qlora

# 如果 GitHub 里的 repo 还没包含 22A 脚本，就从 Drive 备份恢复。
LOCAL_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
if LOCAL_SCRIPT.exists():
    print("Script already present:", LOCAL_SCRIPT)
elif DRIVE_SCRIPT.exists():
    shutil.copy2(DRIVE_SCRIPT, LOCAL_SCRIPT)
    print("Restored script from Drive:", DRIVE_SCRIPT, "->", LOCAL_SCRIPT)
else:
    raise FileNotFoundError(f"Missing helper script in repo and Drive: {SCRIPT_NAME}")

LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Module 22A restore complete.")
print("Manual audit CSV:", MANUAL_AUDIT_CSV)
print("Subset CSV:", SUBSET_CSV)
print("Drive output dir:", DRIVE_OUTPUT_DIR)
```

## 22A.2 Generate cleanup candidates

```python
# Module 22A.2 - 生成 evaluator/data cleanup 清单
from pathlib import Path
import subprocess

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_INPUT_DIR = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/inputs"
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_evaluator_cleanup"

cmd = [
    "python", "scripts/prepare_chartqa_evaluator_cleanup.py",
    "--manual-audit-csv", str(DRIVE_INPUT_DIR / "chartqa_all_wrong_manual_audit_table.csv"),
    "--subset-csv", str(DRIVE_INPUT_DIR / "chartqa_all_wrong_recommended_diagnostic_subset.csv"),
    "--output-dir", "outputs/chartqa_evaluator_cleanup",
    "--drive-output-dir", str(DRIVE_OUTPUT_DIR),
]
subprocess.run(cmd, check=True)
```

## 22A.3 Read the cleanup report and decision notes

```python
# Module 22A.3 - 读取清理报告，给出下一步判断
from pathlib import Path
import json

REPO = Path("/content/qwen25vl-chartqa-qlora")
SUMMARY_JSON = REPO / "outputs/chartqa_evaluator_cleanup/chartqa_evaluator_cleanup_summary.json"
REPORT_MD = REPO / "outputs/chartqa_evaluator_cleanup/chartqa_evaluator_cleanup_report.md"

summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
subset = summary["recommended_subset"]

print("推荐 85 条 subset 样本数:", subset["total"])
print("cleanup candidates:", subset["cleanup_candidate_count"])
print("若仅剔除/修正 exclude_or_fix_reference，有效模型错误数:", subset["effective_model_error_count_if_excluded"])
print("\ncleanup policy counts:")
print(json.dumps(subset["cleanup_policy_counts"], ensure_ascii=False, indent=2))
print("\nissue type counts:")
print(json.dumps(subset["issue_type_counts"], ensure_ascii=False, indent=2))

print("\n报告位置:", REPORT_MD)
print(REPORT_MD.read_text(encoding="utf-8")[:4000])
```

## 22A.4 中文结论

本模块只建立清理清单，不直接改 evaluator，也不重算历史 full-val。

使用口径：

- `exclude_or_fix_reference`：高优先级。先从模型能力增益判断中单独剥离，后续修 reference 或建立 ignore list。
- `normalization_candidate`：中优先级。需要设计 evaluator normalization 规则，并人工抽查。
- `answer_format_manual_review`：中优先级。先确认题目要求 list、sum、顺序无关 list，不能直接当模型失败。

下一步建议：

```text
Module 22B: staged chart-to-table extraction on the same 85-sample subset
```

22B 之前不要扩到 full-val，也不要继续训练新 LoRA。

