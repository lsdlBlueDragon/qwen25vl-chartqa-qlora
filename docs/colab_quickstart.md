# Colab Quickstart

## Goal

Use Colab GPU for model download, baseline inference, ChartQA processing, QLoRA training, and evaluation.

Local Windows remains a non-training smoke-test environment.

## Startup Cells

### 1. Clone Repository

```bash
git clone https://github.com/lsdlBlueDragon/qwen25vl-chartqa-qlora.git
cd qwen25vl-chartqa-qlora
```

If the repository already exists:

```bash
cd qwen25vl-chartqa-qlora
git pull
```

### 2. Optional Domestic Hugging Face Mirror

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Use this only when official Hugging Face downloads are slow or unstable.

### 3. Install Dependencies

Colab usually has a compatible PyTorch build. Avoid reinstalling torch unless needed.

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If dependency conflicts appear, install only missing project packages first:

```bash
pip install trl peft qwen-vl-utils gradio datasets accelerate evaluate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Recommended Drive root:

```python
PROJECT_DRIVE = "/content/drive/MyDrive/qwen25vl-chartqa-qlora"
```

### 5. Environment Check

```bash
python scripts/env_check.py --output outputs/env_check_colab.json
```

### 6. Baseline Single Image

Upload or place one chart image at `/content/example_chart.png`, then run:

```bash
python scripts/run_baseline_image.py \
  --image /content/example_chart.png \
  --question "What is the highest value shown in the chart?" \
  --output outputs/baseline_single.jsonl \
  --load-in-4bit
```

## Checkpoint and Cache Policy

Use Drive for:

- Hugging Face cache;
- ChartQA cache;
- temporary checkpoints;
- large logs;
- exported adapters before uploading to HF Hub.

Do not commit these files to GitHub.

## First Colab Success Criteria

The first Colab run is successful when:

1. repository clones;
2. dependencies install;
3. `env_check.py` detects GPU;
4. one-image baseline inference writes JSONL.

