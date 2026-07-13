# Module 23C - normalization v2 and targeted prompt ablation

This module has two parts:

1. evaluator-only normalization v2 on existing predictions;
2. small targeted prompt ablation on the 28 true-hard samples from Module 23B.

It does **not** run full validation and does **not** train a new adapter. The prompt ablation is intentionally small: routed mode runs one targeted prompt for each true-hard sample, so the default run is 28 generations.

Long-running prediction outputs are append-only JSONL, restored from Drive when available, and skipped on rerun by `(sample_index, prompt_name)`.

## Dependencies, reconnect, and resume rules

Run this module only after the repo is available at `/content/qwen25vl-chartqa-qlora` and Drive is mounted at `/content/drive/MyDrive`.

Recommended fresh-runtime order:

```text
1. Run the base notebook dependency/setup cells if the runtime is fresh.
2. Run 23C.1 to restore scripts, subset, 23A/23B inputs, and the hardmix adapter.
3. Run 23C.2 for evaluator-only normalization v2. This does not need GPU.
4. Run 23C.3 for the 28-sample routed prompt ablation. This needs GPU.
5. Run 23C.4 to read summaries.
```

If Colab disconnects during 23C.3:

```text
1. Do not delete outputs/chartqa_23c_targeted_prompt_ablation/*.jsonl.
2. Reconnect to a GPU runtime.
3. Run dependency/setup cells if imports fail.
4. Run 23C.1 again.
5. Run 23C.3 again with the same arguments.
```

The script restores `targeted_prompt_hardmix_routed.jsonl` from Drive when local output is missing and skips completed `(sample_index, prompt_name)` rows. A rerun should print fewer pending tasks, or `pending: 0` if complete.

Minimal dependency cells for a fresh Colab runtime:

```python
# Module 23C.0-install - dependencies for a fresh Colab runtime
# 中文说明：
# 1. 这个单元只安装依赖，不在同一个进程里 import peft。
# 2. Pillow 在 Colab 中原地升级后容易出现 PIL 新旧文件混用，例如：
#    ImportError: cannot import name '_Ink' from 'PIL._typing'
# 3. 因此本单元执行完成后，请 Runtime -> Restart runtime，再运行 23C.0-check。

%pip install -q --no-cache-dir --force-reinstall "pillow==10.4.0"
%pip install -q -U --no-cache-dir \
    "transformers>=4.51.0" \
    "accelerate>=1.2.0" \
    "peft>=0.14.0" \
    "bitsandbytes>=0.45.0" \
    "qwen-vl-utils>=0.0.8" \
    "tqdm>=4.66.0" \
    "pandas>=2.2.0"

print("Dependency install finished.")
print("如果本单元刚刚安装/升级了 Pillow：请现在重启 runtime，然后从 23C.0-check 继续。")
print("Colab 菜单：Runtime -> Restart runtime")
```

```python
# Module 23C.0-check - post-restart import check
# 中文说明：这个单元必须在重启 runtime 之后运行，用来确认依赖环境干净可用。
import torch
import transformers
import peft
import qwen_vl_utils
from PIL import Image, ImageText

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("pillow:", Image.__version__)
print("qwen_vl_utils import: ok")
```

Checkpoint status cell:

```python
# Module 23C.checkpoint - inspect routed prompt-ablation progress
from pathlib import Path
import json

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

local_pred = REPO / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed.jsonl"
drive_pred = DRIVE_ROOT / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed.jsonl"
metrics = REPO / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed_metrics.json"

for path in [local_pred, drive_pred, metrics]:
    print(path, "exists=", path.exists(), "size=", path.stat().st_size if path.exists() else None)

if local_pred.exists():
    rows = [json.loads(line) for line in local_pred.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = sorted({(int(row["sample_index"]), row["prompt_name"]) for row in rows})
    print("completed predictions:", len(done))
    print("completed sample indices:", sorted({idx for idx, _ in done}))

if metrics.exists():
    print(metrics.read_text(encoding="utf-8")[:3000])
```

## 23C.1 Restore scripts, inputs, and adapter

```python
# Module 23C.1 - restore scripts, inputs, and adapter
from pathlib import Path
import shutil

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print("Drive mount skipped or unavailable:", exc)

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_SCRIPT_DIR = DRIVE_ROOT / "scripts_module23c"
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_23c_targeted_prompt_ablation"
DRIVE_NORM_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_23c_normalization_v2"

SCRIPT_NAMES = [
    "run_chartqa_23c_normalization_v2.py",
    "run_chartqa_23c_targeted_prompt_ablation.py",
]

LOCAL_SUBSET_JSONL = REPO / "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl"
DRIVE_SUBSET_JSONL = DRIVE_ROOT / "outputs/chartqa_all_wrong_diagnostics/data/chartqa_all_wrong_diagnostic_subset_85.jsonl"

LOCAL_23A_PER_PRED = REPO / "outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv"
DRIVE_23A_PER_PRED = DRIVE_ROOT / "outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv"

LOCAL_23B_REVIEW = REPO / "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv"
DRIVE_23B_REVIEW = DRIVE_ROOT / "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv"

HARDMIX_ADAPTER = REPO / "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
DRIVE_HARDMIX_ADAPTER = DRIVE_ROOT / "outputs/adapters/chartqa_qlora_hardmix1k_steps100"

for path in [REPO, DRIVE_ROOT, DRIVE_SCRIPT_DIR, DRIVE_SUBSET_JSONL, DRIVE_23A_PER_PRED, DRIVE_23B_REVIEW]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

%cd /content/qwen25vl-chartqa-qlora

for script_name in SCRIPT_NAMES:
    local_script = REPO / "scripts" / script_name
    drive_script = DRIVE_SCRIPT_DIR / script_name
    local_script.parent.mkdir(parents=True, exist_ok=True)
    if local_script.exists():
        print("Script already present:", local_script)
    elif drive_script.exists():
        shutil.copy2(drive_script, local_script)
        print("Restored script from Drive:", drive_script, "->", local_script)
    else:
        raise FileNotFoundError(f"Missing helper script in repo and Drive: {script_name}")

for local_path, drive_path in [
    (LOCAL_SUBSET_JSONL, DRIVE_SUBSET_JSONL),
    (LOCAL_23A_PER_PRED, DRIVE_23A_PER_PRED),
    (LOCAL_23B_REVIEW, DRIVE_23B_REVIEW),
]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        shutil.copy2(drive_path, local_path)
        print("Restored input:", local_path)

def copytree_if_needed(src: Path, dst: Path) -> None:
    if dst.exists():
        print("Adapter already present:", dst)
        return
    if not src.exists():
        raise FileNotFoundError(f"Missing adapter source: {src}")
    shutil.copytree(src, dst)
    print("Restored adapter:", dst)

copytree_if_needed(DRIVE_HARDMIX_ADAPTER, HARDMIX_ADAPTER)

DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_NORM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Module 23C restore complete.")
print("23A per-prediction:", LOCAL_23A_PER_PRED)
print("23B targeted review:", LOCAL_23B_REVIEW)
print("Hardmix adapter:", HARDMIX_ADAPTER)
```

## 23C.2 Run normalization v2 evaluator-only

```python
# Module 23C.2 - evaluator-only normalization v2
from pathlib import Path
import subprocess
import shutil

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

cmd = [
    "python", "scripts/run_chartqa_23c_normalization_v2.py",
    "--per-prediction-csv", "outputs/chartqa_23a_cleanup_normalization/chartqa_23a_normalization_per_prediction.csv",
    "--targeted-review-csv", "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv",
    "--output-dir", "outputs/chartqa_23c_normalization_v2",
    "--report-md", "docs/experiments/chartqa_23c_normalization_v2_2026-07-03.md",
]
subprocess.run(cmd, check=True)

drive_out = DRIVE_ROOT / "outputs/chartqa_23c_normalization_v2"
drive_out.mkdir(parents=True, exist_ok=True)
for path in (REPO / "outputs/chartqa_23c_normalization_v2").glob("*"):
    if path.is_file():
        shutil.copy2(path, drive_out / path.name)
report = REPO / "docs/experiments/chartqa_23c_normalization_v2_2026-07-03.md"
if report.exists():
    shutil.copy2(report, drive_out / report.name)
print("Normalization v2 outputs copied to:", drive_out)
```

## 23C.3 Run routed targeted prompt ablation

```python
# Module 23C.3 - small routed targeted prompt ablation on 28 true-hard samples
from pathlib import Path
import subprocess
import torch

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the targeted prompt ablation cell.")

cmd = [
    "python", "scripts/run_chartqa_23c_targeted_prompt_ablation.py",
    "--subset-jsonl", "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl",
    "--targeted-review-csv", "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv",
    "--output-dir", "outputs/chartqa_23c_targeted_prompt_ablation",
    "--drive-output-dir", str(DRIVE_ROOT / "outputs/chartqa_23c_targeted_prompt_ablation"),
    "--adapter-path", "outputs/adapters/chartqa_qlora_hardmix1k_steps100",
    "--adapter-name", "hardmix",
    "--prompt-policy", "routed",
    "--max-pixels", "802816",
    "--max-new-tokens", "96",
    "--sync-every", "1",
]
subprocess.run(cmd, check=True)
```

## 23C.4 Read summaries

```python
# Module 23C.4 - read normalization and prompt-ablation summaries
from pathlib import Path
import json

REPO = Path("/content/qwen25vl-chartqa-qlora")

paths = [
    REPO / "outputs/chartqa_23c_normalization_v2/chartqa_23c_normalization_v2_summary.json",
    REPO / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed_metrics.json",
]

for path in paths:
    print("\n==", path, "==")
    if path.exists():
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)[:5000])
    else:
        print("Not found yet.")
```
