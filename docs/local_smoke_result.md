# Local Smoke Check Result

Command:

```powershell
& 'D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe' scripts\env_check.py --output outputs\env_check_local.json
```

Result summary:

- Python executable: `D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe`
- Python version: 3.11.13
- Torch: installed, importable, `2.8.0+cu129`
- CUDA: available
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU`
- VRAM: about 7.96 GB
- Missing local packages:
  - `trl`
  - `qwen-vl-utils`
  - `gradio`

Interpretation:

- The local machine is suitable for non-training smoke checks.
- Do not run QLoRA training locally.
- Before local Gradio or Qwen2.5-VL smoke inference, install the missing lightweight dependencies from a domestic pip source if possible.
- Full model download, ChartQA download, training, and full evaluation should run on Colab GPU.

Note:

- The command produced a complete JSON report but exceeded the 30-second shell timeout. This appears to be slow dependency import or shutdown behavior, not training or model download.
- Full report path: `outputs/env_check_local.json`

