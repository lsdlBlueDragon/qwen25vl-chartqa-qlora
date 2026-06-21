# 本地路线、要求、注意事项

## 选定路线

采用推荐路线：

`Qwen2.5-VL-3B-Instruct -> ChartQA -> QLoRA -> Gradio -> Hugging Face GPU Space`

本地职责：
- 环境检查；
- 代码编辑；
- 小样本数据格式检查；
- 非训练 smoke test；
- Gradio UI 骨架调试；
- 必要时做 1 条样本推理。

Colab 职责：
- 模型下载；
- ChartQA 下载；
- QLoRA smoke training；
- 正式训练；
- 评估；
- adapter 上传。

Hugging Face Space 职责：
- Gradio demo 在线部署；
- 加载 base model + adapter；
- 提供公开演示链接。

## 本地禁止事项

本地 8GB 显卡不用于：

- full ChartQA 训练；
- QLoRA 正式训练；
- 大规模验证集评估；
- 长时间 GPU 任务；
- 反复下载大模型权重。

如果本地推理 OOM，直接转 Colab，不在本地硬调显存。

## 国内源优先策略

### pip

优先使用国内 PyPI 镜像：

```powershell
& 'D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe' -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果某些包在清华源同步不及时，再临时切换到：

```powershell
& 'D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe' -m pip install <package> -i https://mirrors.aliyun.com/pypi/simple/
```

注意：`torch`、`torchvision`、`bitsandbytes`、`flash-attn` 与 CUDA 强相关。Colab 通常已有合适的 torch，优先不要在 Colab 盲目重装 torch。

### Hugging Face 模型和数据

首选官方 Hugging Face 源，保证模型和数据来源可解释。

如果国内网络不稳定，可以在 Colab 或本地临时设置镜像端点：

Colab notebook cell：

```python
%env HF_ENDPOINT=https://hf-mirror.com
```

Windows PowerShell：

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
```

注意：镜像只解决下载问题。实验报告里仍记录官方 repo id，例如 `Qwen/Qwen2.5-VL-3B-Instruct` 和 `HuggingFaceM4/ChartQA`。

### ModelScope 备选

如果 Hugging Face 下载长期不稳定，可以用 ModelScope 作为模型下载备选，但训练和评估代码仍保持 HF repo id 兼容。

要求：
- 记录实际下载源；
- 记录模型 commit 或快照版本；
- 不混用多个来源的权重文件。

## 环境要求

本地：
- Windows；
- Python 使用 `D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe`；
- 只跑 `scripts/env_check.py` 和轻量 smoke test。

Colab：
- 优先 L4/A100/H100；
- T4 可用于 smoke training，但 batch、视觉分辨率和梯度累积要保守；
- checkpoint 存 Google Drive；
- 代码从 GitHub pull。

Space：
- Gradio SDK；
- GPU hardware；
- Secrets 存 `HF_TOKEN`；
- 不把 token 写进代码或 notebook 输出。

## 注意事项

1. 先跑 100 条 QLoRA smoke training，再跑 1k/5k/full。
2. 每次实验保存 config、commit hash、log、predictions、metrics。
3. 训练脚本和评估脚本分离，避免评估结果依赖训练过程。
4. adapter 缺失时 app 要能给出明确提示。
5. 所有大文件默认不进 git。
6. 先验证 Space 能加载 3B 4-bit base，再接 adapter。
7. 如果部署不稳定，优先降低 `max_pixels`，再升级 GPU。
