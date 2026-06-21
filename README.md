# Qwen2.5-VL ChartQA QLoRA Project

End-to-end chart question answering project based on:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `HuggingFaceM4/ChartQA`
- QLoRA / LoRA fine-tuning
- Gradio demo
- Hugging Face Space deployment

## Project Goal

Build a resume-ready AI engineering project that demonstrates:

1. multimodal baseline inference,
2. ChartQA data preparation,
3. low-memory QLoRA fine-tuning on Colab GPU,
4. reproducible evaluation,
5. Gradio demo packaging,
6. Hugging Face Space deployment,
7. experiment notes and interview-ready analysis.

Local machine usage is limited to non-training smoke tests. Training and full evaluation should run on Colab GPU.

## Current Status

- Planning document: `docs/superpowers/specs/2026-06-21-qwen25vl-chartqa-roadmap.md`
- Task breakdown: `docs/tasks_recommended_route.md`
- Local environment guide: `docs/local_route_requirements_notes.md`
- GitHub/Colab/Drive workflow: `docs/github_colab_drive_workflow.md`
- Environment check script: `scripts/env_check.py`

## Local Smoke Test

Use the local Python environment specified in `AGENTS.md`:

```powershell
& 'D:\ProgramData\anaconda3\envs\torch_tf_cuda129\python.exe' scripts\env_check.py --output outputs\env_check_local.json
```

This only checks environment visibility. It does not download models, load ChartQA, or train.
