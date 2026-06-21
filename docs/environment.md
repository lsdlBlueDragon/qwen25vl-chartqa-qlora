# Environment Definition

## Runtime Roles

| Environment | Role | Training Allowed |
| --- | --- | --- |
| Local Windows 8GB GPU | smoke test, code editing, UI skeleton | No |
| Colab GPU | QLoRA training, evaluation, adapter export | Yes |
| Hugging Face Space GPU | public Gradio demo | No |

## Local Check

```powershell
& 'D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe' scripts\env_check.py --output outputs\env_check_local.json
```

The script checks package installation, importability, Python runtime, torch, CUDA visibility, and GPU memory.

## Dependency Installation

Prefer domestic pip mirror when installing locally:

```powershell
& 'D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe' -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Do not install training-only dependencies repeatedly on local Windows if Colab will run training.

## Required Secrets

Do not commit secrets.

Expected names:

- `HF_TOKEN`
- optional `WANDB_API_KEY`

