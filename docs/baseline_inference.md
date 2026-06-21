# Baseline Inference

## Purpose

This module runs Qwen2.5-VL-3B-Instruct on one chart image and one question.

It is the first executable step before ChartQA batch evaluation and QLoRA training.

## Local Policy

Local machine is for non-training smoke tests only.

Allowed locally:

- `--dry-run`;
- syntax/import checks;
- one-image inference only if VRAM allows.

Not allowed locally:

- ChartQA full download;
- batch evaluation;
- QLoRA training.

## Dry Run

```powershell
& 'D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe' scripts\run_baseline_image.py --image app\examples\placeholder.png --question "What is the maximum value?" --dry-run
```

The image path does not need to exist for `--dry-run`.

## Colab Single Image Run

After cloning the repo and installing dependencies:

```bash
python scripts/run_baseline_image.py \
  --image /content/example_chart.png \
  --question "What is the highest value shown in the chart?" \
  --output outputs/baseline_single.jsonl \
  --load-in-4bit
```

If Hugging Face download is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Keep the official model id in config:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

## Output

The script writes JSONL with:

- question;
- answer;
- model id;
- optional adapter path;
- latency;
- generation config;
- image path.

## Verification

Successful baseline verification means:

1. model loads on Colab GPU;
2. one chart image returns a concise answer;
3. output JSONL is created;
4. no training code is invoked.

