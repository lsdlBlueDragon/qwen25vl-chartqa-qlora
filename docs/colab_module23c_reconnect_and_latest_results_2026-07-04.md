# Module 23C-SUPPLEMENT - dependencies, reconnect, checkpoint, and latest results

This supplement fixes the missing operational details for Module 23C. It does not replace the executed 23C cells above.

## Latest 23C Results Read From The Notebook

The latest notebook run completed:

```text
23C.1 restore: executed
23C.2 normalization v2: executed
23C.3 routed targeted prompt ablation: returncode=0
23C.4 summary read: executed
```

Normalization v2:

```text
before v2: 30/67 = 44.78%
after v2:  34/67 = 50.75%
gain: +4
recovered: 18, 241, 648, 816
```

Routed targeted prompt ablation on 28 true-hard samples:

```text
total predictions: 28
unique samples: 28
oracle recovered: 1
recovered index: 344
```

By prompt:

```text
legend_table_prompt: 0/5
multi_answer_prompt: 0/3
operand_table_prompt: 0/15
range_count_prompt: 0/3
spatial_locator_prompt: 1/2, recovered 344
```

Interpretation:

```text
The first routed prompt ablation is mostly negative. The only clear gain is spatial_locator_prompt on sample 344.
This means simple prompt routing is not enough for most hard failures; the remaining bottleneck is structured extraction / localization / operand correctness, not just final-answer wording.
```

## Fresh Runtime Dependency Cell

Run `23C.0-install` after reconnecting to a fresh Colab runtime if imports fail or
if the runtime was reset. Pillow is force-reinstalled separately because Colab can
keep stale `PIL` modules in memory after an in-place upgrade. Restart the runtime
before running the import check.

```python
# Module 23C.0-install - dependencies for a fresh Colab runtime
# 中文说明：
# 1. 这个单元只安装依赖，不在同一个进程里 import peft。
# 2. 如果出现 PIL._typing / _Ink 之类错误，根因通常是 Pillow 原地升级后未重启。
# 3. 本单元执行完成后，请 Runtime -> Restart runtime，再运行 23C.0-check。

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

## Reconnect And Resume Rules

If Colab disconnects during `23C.3`:

```text
1. Reconnect to a GPU runtime.
2. Run 23C.0-install only if dependencies/imports fail, then restart runtime.
3. Run 23C.0-check.
4. Run 23C.1 to restore scripts, inputs, and adapter.
5. Run 23C.checkpoint to inspect completed rows.
6. Run 23C.3 again with the same arguments.
7. Run 23C.4 to read summaries.
```

Do not delete:

```text
outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed.jsonl
```

The script restores the JSONL from Drive if local output is missing and skips completed `(sample_index, prompt_name)` rows. A successful resume should print fewer pending tasks or `pending: 0`.

## Checkpoint Inspection Cell

```python
# Module 23C.checkpoint - inspect routed prompt-ablation progress
from pathlib import Path
import json

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

local_pred = REPO / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed.jsonl"
drive_pred = DRIVE_ROOT / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed.jsonl"
local_eval = REPO / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed_evaluated.jsonl"
metrics = REPO / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed_metrics.json"
drive_metrics = DRIVE_ROOT / "outputs/chartqa_23c_targeted_prompt_ablation/targeted_prompt_hardmix_routed_metrics.json"

for path in [local_pred, drive_pred, local_eval, metrics, drive_metrics]:
    print(path, "exists=", path.exists(), "size=", path.stat().st_size if path.exists() else None)

if local_pred.exists():
    rows = [json.loads(line) for line in local_pred.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = sorted({(int(row["sample_index"]), row["prompt_name"]) for row in rows})
    print("completed predictions:", len(done))
    print("completed sample indices:", sorted({idx for idx, _ in done}))

if metrics.exists():
    print("\nmetrics:")
    print(metrics.read_text(encoding="utf-8")[:5000])
```

## Exact Resume Command For Routed Run

```python
# Module 23C.resume - rerun routed targeted prompt ablation safely
from pathlib import Path
import subprocess
import torch

REPO = Path("/content/qwen25vl-chartqa-qlora")
DRIVE_ROOT = Path("/content/drive/MyDrive/qwen25vl-chartqa-qlora")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this cell. Switch Colab runtime to GPU.")

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

## Next Diagnostic Recommendation

Do not expand to full-val from this routed prompt result. The gain is only `1/28`, and the only winning family is spatial localization.

Recommended next action:

```text
Run all-prompts only on a smaller priority subset, not all 28 at once:
326, 344, 281, 291, 467, 529, 675, 781, 877, 901
```

The goal should be to identify whether any prompt family helps a specific failure type before spending more GPU time.
