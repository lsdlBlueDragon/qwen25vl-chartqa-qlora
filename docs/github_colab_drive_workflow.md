# GitHub + Google Drive + Colab Workflow

## Recommendation

Create the GitHub repository first.

Use:

- GitHub for code, configs, notebooks, docs, small examples, issue tracking, and README;
- Google Drive for Colab caches, intermediate checkpoints, downloaded datasets, and temporary run outputs;
- Hugging Face Hub for final adapter/model artifacts and Space deployment.

Do not use Google Drive as the main source-code location. Drive sync can create duplicated files, hidden conflicts, and unclear version history.

## Why GitHub First

GitHub gives this project:

1. clean commit history;
2. easy Colab `git clone`;
3. reproducible experiment configs;
4. a public resume link;
5. clear separation between code and large artifacts.

Drive is still useful, but only as a storage backend for large or temporary files.

## What Goes Where

| Artifact | Location | Reason |
| --- | --- | --- |
| source code | GitHub | versioned and reviewable |
| configs | GitHub | every experiment should be reproducible |
| docs and README | GitHub | resume/interview visibility |
| notebooks | GitHub | thin orchestration only |
| ChartQA cache | Google Drive or Colab cache | too large for Git |
| model checkpoints | Google Drive during training | large temporary artifacts |
| final LoRA adapter | Hugging Face model repo | easy Space loading |
| Gradio Space app | Hugging Face Space repo | public deployment |
| logs/metrics summaries | GitHub if small, Drive if large | preserve key results without bloating repo |

## Create a GitHub Repo

### Option A: GitHub Website

1. Open GitHub.
2. Click `New repository`.
3. Repository name suggestion: `qwen25vl-chartqa-qlora`.
4. Choose visibility:
   - public: best for resume once no secrets or private data are included;
   - private: safer while developing, can be made public later.
5. Do not initialize with README, `.gitignore`, or license because this folder already has those project files.
6. Create the repository.
7. Copy the HTTPS remote URL.

Then run locally:

```powershell
git init
git add .
git commit -m "Initialize Qwen2.5-VL ChartQA QLoRA project"
git branch -M main
git remote add origin https://github.com/<your-user>/qwen25vl-chartqa-qlora.git
git push -u origin main
```

### Option B: GitHub CLI

If `gh` is installed and logged in:

```powershell
git init
git add .
git commit -m "Initialize Qwen2.5-VL ChartQA QLoRA project"
git branch -M main
gh repo create qwen25vl-chartqa-qlora --public --source . --remote origin --push
```

Use `--private` instead of `--public` if you want to keep it private first.

## Git LFS

This project should normally avoid committing model weights.

`.gitattributes` includes LFS patterns for large model files as a safety net:

- `*.safetensors`
- `*.bin`
- `*.pt`
- `*.pth`

If Git LFS is not installed, do not add these files to Git. Keep them in Google Drive during training and upload final adapters to Hugging Face Hub.

## Google Drive Layout

Recommended Drive directory:

```text
MyDrive/
  qwen25vl-chartqa-qlora/
    cache/
      huggingface/
      datasets/
    checkpoints/
      smoke/
      run_001/
    logs/
    exports/
```

Colab should mount Drive and set cache/output paths explicitly:

```python
from google.colab import drive
drive.mount("/content/drive")

PROJECT_DRIVE = "/content/drive/MyDrive/qwen25vl-chartqa-qlora"
HF_HOME = f"{PROJECT_DRIVE}/cache/huggingface"
```

## Colab Startup Flow

Each new Colab runtime should start from GitHub:

```bash
git clone https://github.com/<your-user>/qwen25vl-chartqa-qlora.git
cd qwen25vl-chartqa-qlora
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python scripts/env_check.py --output outputs/env_check_colab.json
```

If Hugging Face download is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Keep the official model and dataset IDs in configs even when using a mirror endpoint.

## Daily Development Loop

Local:

```powershell
git status
git add <changed-files>
git commit -m "<short message>"
git push
```

Colab:

```bash
git pull
python scripts/env_check.py
```

If code changes are made in Colab:

1. inspect `git diff`;
2. commit only intentional code/config/doc changes;
3. do not commit checkpoints, caches, tokens, or downloaded datasets.

## Secret Handling

Never commit:

- Hugging Face token;
- WandB token;
- Google Drive private paths with sensitive information;
- Space secrets.

Use:

- local `.env`, ignored by Git;
- Colab secrets or runtime variables;
- Hugging Face Space Secrets.

## Practical Decision

Start with GitHub now.

Use Drive after the first Colab run, when there are actual caches and checkpoints to store.

