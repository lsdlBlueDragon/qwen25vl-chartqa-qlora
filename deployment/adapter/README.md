---
base_model: Qwen/Qwen2.5-VL-3B-Instruct
library_name: peft
pipeline_tag: visual-question-answering
tags:
  - qwen2.5-vl
  - chartqa
  - qlora
  - peft
  - vision-language
datasets:
  - HuggingFaceM4/ChartQA
language:
  - en
license: apache-2.0
---

# Qwen2.5-VL-3B ChartQA Hardmix QLoRA

This PEFT adapter fine-tunes `Qwen/Qwen2.5-VL-3B-Instruct` for concise chart question answering. It is the deployment candidate from the associated ChartQA engineering project.

## Evaluation

Evaluation uses the complete 1,920-sample ChartQA validation split and the project's frozen relaxed evaluator.

| Model | Exact | Relaxed | Human relaxed | Machine relaxed |
|---|---:|---:|---:|---:|
| 4-bit base model | 65.73% | 75.94% | 68.96% | 82.92% |
| Hardmix QLoRA | 69.32% | **77.86%** | 72.19% | 83.54% |

The adapter improves relaxed accuracy by 1.93 percentage points. Most of the improvement occurs on human-authored questions. Results are from one training seed; small differences between adapter variants should not be interpreted as statistically established rankings.

## Intended use

- English chart question answering;
- demonstration and educational use;
- comparison with the unadapted Qwen2.5-VL-3B model.

The model is not intended for high-stakes extraction, financial decisions, or accessibility-critical chart interpretation without human review.

## Loading

```python
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration

base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "<HUGGING_FACE_ADAPTER_REPO>")
```

Use the base model processor and the prompt implemented in `src/infer.py` in the project repository.

## Training summary

- Method: QLoRA / PEFT LoRA
- Training subset: 1,000-sample hardmix subset
- Training steps: 100
- Adapter size: approximately 74.4 MB
- Base model inference: 4-bit quantized in the reported evaluation

Exact training configuration should be copied from the archived adapter configuration and project experiment record before publication.

## Limitations

The remaining failures are concentrated in legend binding, visual localization, operand extraction, range aggregation, multi-answer enumeration, and multi-step calculation. Increasing image resolution, using a tested 7B configuration, and targeted prompt routing did not provide stable improvements in the project experiments.

ChartQA also contains some ambiguous or inconsistent references. The headline result uses the frozen project evaluator; post-hoc normalization experiments are reported only as diagnostics.

## Reproducibility

The final run registry is `configs/final_runs.yaml`. Before publishing this card, replace the placeholder repository ID and add:

- public project repository URL;
- adapter repository revision;
- adapter weight SHA-256;
- final package versions;
- training seed and complete hyperparameters.
