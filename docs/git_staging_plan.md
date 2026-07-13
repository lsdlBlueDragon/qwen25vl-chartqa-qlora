# Git staging plan

Prepared: 2026-07-13

No files were staged or committed while preparing this plan. Use explicit path groups; do not use `git add .`.

## Pre-staging verification

Completed in the current workspace:

- Python syntax compilation for `src`, `scripts`, `app`, and `tests`;
- pytest suite: 9 passed;
- Gradio app dry-run;
- final run YAML parse;
- Hugging Face adapter and Space metadata parse;
- untracked file size/type triage.

Known non-blocking warnings:

- the local environment has a `requests` dependency compatibility warning;
- Gradio 5 warns that SVG example watermarking is unsupported;
- Gradio 5 emits an internal Gradio 6 deprecation warning for `gr.Examples`;
- Git reports expected LF-to-CRLF conversion for `.gitignore` on Windows.

## Recommended commit groups

### Commit 1: preserve completed experiment implementation and evidence

Include:

```text
qwen25vl_3b_chartqa_qlora.ipynb
data/diagnostics/
scripts/append_notebook_analysis_cell.py
scripts/audit_chartqa_all_runs_wrong.py
scripts/build_chartqa_all_wrong_review_pack.py
scripts/integrate_colab_module_md.py
scripts/prepare_chartqa_all_wrong_subset.py
scripts/prepare_chartqa_evaluator_cleanup.py
scripts/run_chartqa_23a_cleanup_normalization_ablation.py
scripts/run_chartqa_23b_hard_failure_diagnostics.py
scripts/run_chartqa_23c_normalization_v2.py
scripts/run_chartqa_23c_targeted_prompt_ablation.py
scripts/run_chartqa_24a_structured_hard_ablation.py
scripts/run_chartqa_staged_extraction_diagnostic.py
scripts/run_chartqa_staged_extraction_quality_audit.py
scripts/run_chartqa_structured_extraction_diagnostic.py
scripts/run_chartqa_subset_ablation.py
scripts/summarize_chartqa_all_wrong_diagnostics.py
docs/colab_module*.md
docs/experiments/*.md
```

Suggested message:

```text
Add ChartQA diagnostic experiments and notebook record
```

24A code and instructions are included as planned work; reports must continue to state that no verified 24A result exists.

### Commit 2: freeze project review and experiment policy

Include:

```text
.gitignore
configs/final_runs.yaml
docs/project_review_2026-07-13.md
docs/project_execution_plan.md
docs/project_handoff.md
docs/git_exclusion_policy.md
docs/git_staging_plan.md
```

Suggested message:

```text
Freeze ChartQA results and project execution plan
```

### Commit 3: add tested Gradio demo and release preparation

Include:

```text
README.md
requirements-dev.txt
app/README.md
app/app.py
app/examples/quarterly_sales.svg
tests/test_app.py
tests/test_cli_dry_runs.py
tests/test_prepare_chartqa_sft.py
deployment/
docs/huggingface_release_checklist.md
```

Suggested message:

```text
Add tested ChartQA demo and Hugging Face packaging
```

## Post-commit verification

After all approved commits:

1. confirm `git status --short` is empty;
2. create a temporary clean worktree from `HEAD`;
3. install `requirements-dev.txt` in the verification environment;
4. run `python -m pytest -q`;
5. run `python app/app.py --dry-run`;
6. parse `configs/final_runs.yaml` and both Hugging Face metadata blocks;
7. remove the temporary worktree only after confirming its resolved path is inside the intended workspace or explicitly approved location.

Do not attempt real 3B inference during the non-GPU clean-checkout gate.
