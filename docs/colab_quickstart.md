# Colab Quickstart

## Goal

Use Colab GPU for model download, baseline inference, ChartQA processing, QLoRA training, and evaluation.

Local Windows remains a non-training smoke-test environment.

All Colab commands below are written as notebook cells. Paste them directly into `.ipynb` cells.

## Startup Cells

### 1. Clone Repository

```python
!git clone https://github.com/lsdlBlueDragon/qwen25vl-chartqa-qlora.git
%cd qwen25vl-chartqa-qlora
```

If the repository already exists:

```python
%cd /content/qwen25vl-chartqa-qlora
!git pull
```

### 2. Install Dependencies

Colab usually has a compatible PyTorch build. Avoid reinstalling torch unless needed.

```python
!pip install -r requirements.txt
```

If dependency conflicts appear, install only missing project packages first:

```python
!pip install trl peft qwen-vl-utils gradio datasets accelerate evaluate bitsandbytes
```

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Recommended Drive root:

```python
PROJECT_DRIVE = "/content/drive/MyDrive/qwen25vl-chartqa-qlora"
```

### 4. Optional Environment Check

```python
!python scripts/env_check.py --output outputs/env_check_colab.json
```

This is optional. In normal Colab work, it is enough to confirm the runtime shows a GPU and then run the baseline/training script.

### 5. Baseline Single Image

Upload or place one chart image at `/content/example_chart.png`, then run:

```python
!python scripts/run_baseline_image.py \
  --image /content/example_chart.png \
  --question "What is the highest value shown in the chart?" \
  --output outputs/baseline_single.jsonl \
  --load-in-4bit
```

Optional: show the JSONL output in the notebook:

```python
from pathlib import Path

print(Path("outputs/baseline_single.jsonl").read_text(encoding="utf-8"))
```

### 6. ChartQA Small-Batch Baseline

After single-image baseline works, run a 5-sample ChartQA baseline:

```python
%cd /content/qwen25vl-chartqa-qlora

!python scripts/run_chartqa_baseline.py \
  --n-samples 5 \
  --split train \
  --output outputs/chartqa_baseline_5.jsonl \
  --drive-output-dir /content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/chartqa_baseline \
  --load-in-4bit
```

This cell performs inference, local JSONL saving, Drive backup, and a small exact-match summary in one run.

### 7. Evaluate Prediction JSONL

```python
%cd /content/qwen25vl-chartqa-qlora

!python scripts/evaluate_predictions.py \
  --predictions outputs/chartqa_val_baseline_20.jsonl \
  --metrics-output outputs/chartqa_val_baseline_20_metrics.json \
  --drive-metrics-dir /content/drive/MyDrive/qwen25vl-chartqa-qlora/outputs/metrics
```

### 8. Prepare ChartQA SFT Data

```python
%cd /content/qwen25vl-chartqa-qlora

PROJECT_DRIVE = "/content/drive/MyDrive/qwen25vl-chartqa-qlora"

!python scripts/prepare_chartqa_sft.py \
  --split train \
  --n-samples 100 \
  --output data/processed/chartqa_train_sft_100.jsonl \
  --drive-output-dir {PROJECT_DRIVE}/data/processed

!python scripts/prepare_chartqa_sft.py \
  --split val \
  --n-samples 50 \
  --output data/processed/chartqa_val_sft_50.jsonl \
  --drive-output-dir {PROJECT_DRIVE}/data/processed
```

### 9. QLoRA Smoke Training

```python
%cd /content/qwen25vl-chartqa-qlora
!git pull

PROJECT_DRIVE = "/content/drive/MyDrive/qwen25vl-chartqa-qlora"

!python scripts/train_qwen25vl_qlora.py \
  --train-jsonl data/processed/chartqa_train_sft_100.jsonl \
  --eval-jsonl data/processed/chartqa_val_sft_50.jsonl \
  --output-dir outputs/adapters/chartqa_qlora_smoke_100 \
  --drive-output-dir {PROJECT_DRIVE}/outputs/adapters \
  --max-steps 20 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
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
