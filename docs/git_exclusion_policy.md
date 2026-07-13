# Git inclusion and exclusion policy

The repository should contain enough material to understand, test, and reproduce the project without becoming a mirror of Colab, Google Drive, or the Hugging Face cache.

## Commit these files

- source code under `src/`, `scripts/`, and `app/`;
- stable configuration and run manifests under `configs/`;
- focused tests under `tests/`;
- the final notebook when it is intentionally maintained as a project deliverable;
- final experiment reports and small reviewed diagnostic fixtures;
- README, setup, evaluation, deployment, review, plan, and handoff documentation;
- empty directory placeholders such as `.gitkeep`.

Experiment reports are project evidence, not unrelated clutter. Keep the final reports that support a published metric. Consolidate or archive duplicate notebook commentary rather than hiding all `docs/experiments` files with `.gitignore`.

## Never commit these files

- base-model caches and Hugging Face cache internals;
- `*.safetensors`, `*.bin`, `*.pt`, and `*.pth` model weights;
- full datasets, generated image directories, Arrow, or Parquet caches;
- bulk predictions and generated output directories;
- secrets, `.env`, tokens, and account credentials;
- virtual environments, Python caches, test caches, logs, and temporary files;
- local assistant state such as `.claude/`, `.agents/`, and `.codex/`;
- Google Drive placeholder files and conflicted copies.

## Historical handoff policy

Use `docs/project_handoff.md` as the single current handoff. Root-level dated files matching `qwen25vl_chartqa_handoff_*.md` are ignored because they duplicate and sometimes contradict the current state. Existing historical copies may remain outside Git as private working records.

## Large artifact distribution

Publish the selected adapter through a versioned Hugging Face model repository. The repository should record its model ID, revision, file checksum, training configuration, and benchmark result in `configs/final_runs.yaml` or the model card.

The G drive remains a private backup. Local and G drive absolute paths may appear in handoff documentation, but application code and public reproduction commands must not require those paths.

Repository safety backups must not be retained on the project C drive. Use this storage order:

1. `F:\LLM_project_backups\qwen25vl-chartqa-qlora`;
2. `E:\LLM_project_backups\qwen25vl-chartqa-qlora` when F is unavailable.

Every backup must include a SHA-256 manifest. For Git history, create and verify a `git bundle`; for uncommitted work, preserve relative paths in an archive. The repository-local `backups/` ignore rule is only a temporary safety guard and must not be used for long-term storage.

## Before staging changes

Run:

```powershell
git status --short
git check-ignore -v <path>
git diff -- .gitignore
```

Review every new file by category. Do not use broad rules such as `docs/*`, `*.md`, `scripts/*`, or `*.ipynb`; those would hide real project deliverables.

Use `git add` with explicit paths for the first cleanup commit. Do not use `git add .` until the untracked-file triage in the execution plan is complete.

## 2026-07-13 untracked-file triage

The visible untracked set was inspected by path, type, and size. No model weights, cache trees, bulk predictions, or full image datasets were found in that set.

Intended for version control:

- `qwen25vl_3b_chartqa_qlora.ipynb` as the maintained project notebook;
- current `scripts/*.py` experiment and diagnostic implementations;
- `docs/colab_module*.md` as executable Colab workflow records;
- `docs/experiments/*.md` as evidence supporting the final review;
- `data/diagnostics/` as the small reviewed diagnostic fixture (approximately 109 KB);
- the review, plan, handoff, Git policy, and final run registry added on 2026-07-13.

Ignored local-only material:

- `.claude/`;
- root-level dated handoff snapshots;
- cache, output, weight, and Drive-conflict patterns already listed in `.gitignore`.

No existing file was deleted or staged during this triage.
