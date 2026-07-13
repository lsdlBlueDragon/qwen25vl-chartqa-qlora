# Qwen2.5-VL-3B ChartQA QLoRA

An end-to-end chart question answering project built with Qwen2.5-VL-3B, ChartQA, QLoRA, reproducible evaluation, and a Gradio demo.

The best adapter improves relaxed accuracy on the complete 1,920-sample ChartQA validation split from **75.94% to 77.86%**. The largest gain is on human-authored questions.

## Results

All headline numbers use the frozen evaluator in `src/eval_chartqa.py`. Post-hoc normalization experiments are diagnostic only.

| Run | Exact | Relaxed | Human relaxed | Machine relaxed |
|---|---:|---:|---:|---:|
| 4-bit Qwen2.5-VL-3B baseline | 65.73% | 75.94% | 68.96% | 82.92% |
| Standard QLoRA, 100 steps | 68.59% | 77.24% | 71.77% | 82.71% |
| QLoRA, 200 steps | 69.01% | 77.55% | 71.88% | 83.23% |
| Calc/numeric QLoRA | 68.91% | 77.45% | 71.15% | 83.75% |
| **Hardmix QLoRA** | 69.32% | **77.86%** | **72.19%** | 83.54% |
| Rank-16 QLoRA, 250 steps | **69.48%** | 77.66% | 71.98% | 83.33% |

The hardmix adapter is the deployment candidate. The rank-16/250-step adapter is retained as the best exact-accuracy comparison.

The seven-run relaxed oracle reaches 83.07%, but it is a diagnostic upper bound rather than a deployable result.

## What the experiments showed

- QLoRA provides a modest, repeatable improvement over the 3B baseline.
- Human-authored relaxed accuracy improves by 3.23 percentage points; machine-authored accuracy improves by 0.63 points.
- Adapter variants are close: their relaxed scores span only 0.62 percentage points.
- Tested 7B configurations did not beat the comparable 3B result.
- Increasing image resolution and targeted prompt routing did not solve the remaining hard cases.
- Remaining errors concentrate in legend binding, visual localization, operand extraction, range aggregation, multi-answer enumeration, and multi-step calculation.
- Staged structured extraction recovers some difficult cases, but is not reliable enough to replace direct QA.

## Project structure

```text
app/             Gradio application and self-created example chart
configs/         Project configuration and frozen final run registry
data/diagnostics Small reviewed diagnostic fixture; full data is not committed
deployment/      Adapter model card and Hugging Face Space templates
docs/            Setup, evaluation, experiments, review, plan, and handoff
scripts/         Data preparation, training, evaluation, and diagnostics
src/             Shared inference and ChartQA evaluation code
tests/           Non-GPU regression tests
```

## Installation

Python 3.11 and a CUDA GPU environment are recommended for model inference. Full training and full-validation inference were designed for Colab GPU runtimes.

Runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Development and tests:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The local test suite does not download ChartQA or load the 3B model.

## Gradio demo

Build the UI without loading model weights:

```powershell
python app\app.py --dry-run
```

Launch it in a GPU environment:

```powershell
$env:CHARTQA_ADAPTER_PATH = "outputs/adapters/chartqa_qlora_hardmix1k_steps100"
python app\app.py --host 0.0.0.0 --port 7860
```

`CHARTQA_ADAPTER_PATH` may be a local PEFT adapter directory or, after publication, a Hugging Face adapter repository ID.

The UI supports:

- chart image upload;
- concise chart question input;
- base/hardmix model selection;
- answer and generation latency output;
- three built-in questions using the repository-owned `app/examples/quarterly_sales.svg` chart.

Only one model configuration is retained in memory. Switching modes releases the previous model before loading the next one.

Online Hugging Face Space: **not published yet**. Packaging files are prepared under `deployment/space/`.

## Command-line smoke tests

Single-image inference dry-run:

```powershell
python scripts\run_baseline_image.py `
  --image app\examples\quarterly_sales.svg `
  --question "Which region had the highest sales in Q4?" `
  --dry-run
```

ChartQA data preparation dry-run:

```powershell
python scripts\prepare_chartqa_sft.py --split train --n-samples 100 --dry-run
```

ChartQA baseline dry-run:

```powershell
python scripts\run_chartqa_baseline.py --n-samples 5 --dry-run
```

Real training and benchmark inference require the model, dataset access, and a CUDA GPU.

## Training

The QLoRA entry point is:

```powershell
python scripts\train_qwen25vl_qlora.py `
  --train-jsonl data\processed\chartqa_train_sft_1000.jsonl `
  --output-dir outputs\adapters\chartqa_qlora_run
```

Start with the documented smoke workflow before reproducing a larger run. Adapter weights and full datasets are intentionally excluded from Git.

## Evaluation policy

The primary benchmark is the complete ChartQA `val` split with 1,920 samples. The frozen run registry is `configs/final_runs.yaml`.

Keep these evaluation scopes separate:

- early sequential/head val100;
- fixed random seed42 val100;
- complete val1920.

Normalization v2 and evaluator-cleanup results are useful for diagnosing ambiguous labels and equivalent outputs, but they are not used to select the headline model.

## Reproducibility and artifacts

The Git repository contains code, tests, selected reports, configuration, and small diagnostic fixtures. It intentionally excludes:

- base-model caches;
- adapter weights;
- full ChartQA images and processed data;
- bulk predictions and generated experiment outputs;
- credentials and local tool state.

The adapter model card template is in `deployment/adapter/README.md`. Publication and Space deployment remain blocked until a Hugging Face repository ID, credentials, and GPU hardware are selected.

## Limitations

- The best improvement is 1.93 percentage points, not a large benchmark breakthrough.
- Reported adapter runs use one training seed; very small differences between adapters are not statistically established.
- The model remains unreliable on visually dense charts, ambiguous legends, multi-step arithmetic, and multi-answer questions.
- ChartQA contains some ambiguous or inconsistent references.
- A real base/adapter GPU smoke test is still required after the public adapter is uploaded.
- CPU-only Hugging Face Space hardware is not an appropriate deployment target for this 3B VLM workflow.

## Project status and documentation

- Combined local/G-drive review: `docs/project_review_2026-07-13.md`
- Execution plan and completion state: `docs/project_execution_plan.md`
- Current handoff: `docs/project_handoff.md`
- Git inclusion/exclusion policy: `docs/git_exclusion_policy.md`
- Hugging Face release checklist: `docs/huggingface_release_checklist.md`
- Original roadmap: `docs/superpowers/specs/2026-06-21-qwen25vl-chartqa-roadmap.md`

The research and local demo preparation are complete. Public adapter publication and GPU Space deployment are the remaining external milestones.
