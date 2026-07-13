# Module 24A - structured intermediate output ablation

Module 24A tests whether forcing a structured intermediate representation helps the 28 true-hard samples that remained after cleanup and normalization.

It does **not** run full-val and does **not** train a new adapter.

Default run:

```text
28 true-hard samples × 1 routed structured prompt = 28 generations
```

Optional `--prompt-policy all` run:

```text
28 true-hard samples × 5 structured prompts = 140 generations
```

## 24A.0 Dependencies and reconnect rules

Run `24A.0-install` only on a fresh Colab runtime, or after an import error.
Pillow is installed in a separate force-reinstall step because Colab can keep old
`PIL` modules in memory after an in-place upgrade. If this cell installs or
upgrades Pillow, restart the runtime before importing `peft` / `transformers`.

```python
# Module 24A.0-install - dependencies for a fresh Colab runtime
# 中文说明：
# 1. 这个单元只负责安装依赖，不在同一个 Python 进程里 import peft。
# 2. Pillow 在 Colab 中原地升级后容易出现 PIL 新旧文件混用，例如：
#    ImportError: cannot import name '_Ink' from 'PIL._typing'
# 3. 因此本单元执行完成后，请先 Runtime -> Restart runtime，再运行 24A.0-check。

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
print("如果本单元刚刚安装/升级了 Pillow：请现在重启 runtime，然后从 24A.0-check 继续。")
print("Colab 菜单：Runtime -> Restart runtime")
```

```python
# Module 24A.0-check - post-restart import check
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

If Colab disconnects during `24A.2`, reconnect to GPU and run:

```text
24A.0-install only if dependencies are missing, then restart runtime
24A.0-check
24A.1 restore
24A.checkpoint inspect progress
24A.2 resume with same arguments
24A.3 read summary
```

The prediction JSONL is append-only and skips completed `(sample_index, prompt_name)` rows.

## 24A.1 Restore scripts, inputs, and adapter

```python
# Module 24A.1 - restore scripts, inputs, and hardmix adapter
from pathlib import Path
import shutil

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except Exception as exc:
    print("Drive mount skipped or unavailable:", exc)

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")
DRIVE_SCRIPT_DIR = DRIVE_ROOT / "scripts_module24a"
DRIVE_OUTPUT_DIR = DRIVE_ROOT / "outputs/chartqa_24a_structured_hard_ablation"

%cd /content/qwen25vl-chartqa-qlora

SCRIPT_NAME = "run_chartqa_24a_structured_hard_ablation.py"
LOCAL_SCRIPT = REPO / "scripts" / SCRIPT_NAME
DRIVE_SCRIPT = DRIVE_SCRIPT_DIR / SCRIPT_NAME
if LOCAL_SCRIPT.exists():
    print("Script already present:", LOCAL_SCRIPT)
elif DRIVE_SCRIPT.exists():
    LOCAL_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DRIVE_SCRIPT, LOCAL_SCRIPT)
    print("Restored script:", DRIVE_SCRIPT, "->", LOCAL_SCRIPT)
else:
    raise FileNotFoundError(f"Missing 24A script in repo and Drive: {SCRIPT_NAME}")

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
    DRIVE_ROOT / "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv",
    REPO / "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv",
)

HARDMIX_ADAPTER = REPO / "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
DRIVE_HARDMIX_ADAPTER = DRIVE_ROOT / "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
if HARDMIX_ADAPTER.exists():
    print("Adapter already present:", HARDMIX_ADAPTER)
else:
    if not DRIVE_HARDMIX_ADAPTER.exists():
        raise FileNotFoundError(f"Missing hardmix adapter on Drive: {DRIVE_HARDMIX_ADAPTER}")
    shutil.copytree(DRIVE_HARDMIX_ADAPTER, HARDMIX_ADAPTER)
    print("Restored hardmix adapter:", HARDMIX_ADAPTER)

DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("24A restore complete.")
```

## 24A.checkpoint Inspect progress

```python
# Module 24A.checkpoint - inspect structured ablation checkpoint
from pathlib import Path
import json

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

local_pred = REPO / "outputs/chartqa_24a_structured_hard_ablation/structured_24a_hardmix_routed.jsonl"
drive_pred = DRIVE_ROOT / "outputs/chartqa_24a_structured_hard_ablation/structured_24a_hardmix_routed.jsonl"
local_eval = REPO / "outputs/chartqa_24a_structured_hard_ablation/structured_24a_hardmix_routed_evaluated.jsonl"
metrics = REPO / "outputs/chartqa_24a_structured_hard_ablation/structured_24a_hardmix_routed_metrics.json"

for path in [local_pred, drive_pred, local_eval, metrics]:
    print(path, "exists=", path.exists(), "size=", path.stat().st_size if path.exists() else None)

if local_pred.exists():
    rows = [json.loads(line) for line in local_pred.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = sorted({(int(row["sample_index"]), row["prompt_name"]) for row in rows})
    print("completed predictions:", len(done))
    print("completed sample indices:", sorted({idx for idx, _ in done}))

if metrics.exists():
    print(metrics.read_text(encoding="utf-8")[:5000])
```

## 24A.2 Run routed structured ablation

```python
# Module 24A.2 - run routed structured intermediate ablation on 28 true-hard samples
from pathlib import Path
import subprocess
import torch

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for 24A. Switch Colab runtime to GPU.")

cmd = [
    "python", "scripts/run_chartqa_24a_structured_hard_ablation.py",
    "--subset-jsonl", "data/diagnostics/chartqa_all_wrong_diagnostic_subset_85.jsonl",
    "--targeted-review-csv", "outputs/chartqa_23b_hard_failure_diagnostics/chartqa_23b_codex_targeted_review.csv",
    "--output-dir", "outputs/chartqa_24a_structured_hard_ablation",
    "--drive-output-dir", str(DRIVE_ROOT / "outputs/chartqa_24a_structured_hard_ablation"),
    "--adapter-path", "outputs/adapters/chartqa_qlora_hardmix1k_steps100",
    "--adapter-name", "hardmix",
    "--prompt-policy", "routed",
    "--max-pixels", "802816",
    "--max-new-tokens", "256",
    "--sync-every", "1",
]
subprocess.run(cmd, check=True)
```

## 24A.3 Read summary

```python
# Module 24A.3 - read structured ablation summary
from pathlib import Path
import json

REPO = Path("/content/qwen25vl-chartqa-qlora")
metrics = REPO / "outputs/chartqa_24a_structured_hard_ablation/structured_24a_hardmix_routed_metrics.json"
evaluated = REPO / "outputs/chartqa_24a_structured_hard_ablation/structured_24a_hardmix_routed_evaluated.jsonl"

print("metrics:", metrics, "exists=", metrics.exists())
if metrics.exists():
    print(json.dumps(json.loads(metrics.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)[:6000])

if evaluated.exists():
    rows = [json.loads(line) for line in evaluated.read_text(encoding="utf-8").splitlines() if line.strip()]
    print("evaluated rows:", len(rows))
    print("recovered:")
    for row in rows:
        if row.get("eval_normalization_v2_correct"):
            print(row["sample_index"], row["prompt_name"], "answer=", row.get("parsed_final_answer"))
```
