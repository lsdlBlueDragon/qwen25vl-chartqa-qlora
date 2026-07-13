# Project handoff

Last updated: 2026-07-13

## Current state

The ChartQA research result is frozen. Qwen2.5-VL-3B QLoRA improves full-validation relaxed accuracy from 75.94% to 77.86%. The hardmix adapter is the deployment candidate. The steps250/r16/a32 adapter has the best exact accuracy at 69.48%.

The local repository contains the implementation and analysis documents. The G drive contains the complete datasets, adapters, predictions, metrics, and diagnostic outputs. Do not modify or delete G drive artifacts as part of normal repository cleanup.

## Authoritative files

- Review and evidence: `docs/project_review_2026-07-13.md`
- Plan and status: `docs/project_execution_plan.md`
- Frozen run registry: `configs/final_runs.yaml`
- Git policy: `docs/git_exclusion_policy.md`
- Core evaluator: `src/eval_chartqa.py`
- Inference helper: `src/infer.py`
- Training entry point: `scripts/train_qwen25vl_qlora.py`

Older root-level `qwen25vl_chartqa_handoff_YYYY-MM-DD.md` files are historical snapshots. They are ignored for future work and must not override this handoff.

## Artifact locations

G drive root:

```text
G:\我的云端硬盘\qwen25vl-chartqa-qlora
```

Important subdirectories:

```text
outputs/adapters/chartqa_qlora_hardmix1k_steps100
outputs/adapters/chartqa_qlora_train1k_steps250_r16a32
outputs/chartqa_3b_full_val
outputs/chartqa_3b_new_benchmark
outputs/chartqa_all_wrong_diagnostics
outputs/chartqa_23c_normalization_v2
outputs/chartqa_23c_targeted_prompt_ablation
```

All six inspected adapter directories contain a model weight file and configuration files. Full-val has all seven expected runs and no missing run.

## Important cautions

- Do not report 24A as completed. Local code and Colab instructions exist, but no 24A result directory was found in the G drive archive.
- Do not mix sequential val100, random seed42 val100, and full val1920 metrics.
- Do not use normalization v2 as the headline model metric.
- Do not claim tests pass until pytest is installed and actually reports collected tests.
- Do not commit model weights, HF caches, full image datasets, or bulk prediction outputs.
- Do not remove untracked experiment scripts or reports merely to clean `git status`; triage them against the Git policy first.

## Resume point

Continue with the Git diff review described in `docs/project_execution_plan.md`. Do not stage or commit until the user authorizes that Git operation. P5.2 requires a clean committed state. Publishing and Space deployment require explicit user credentials and repository decisions.

## Handoff update rule

Update this file in place after each meaningful work block. Record the new status in the execution plan, update artifact paths if they change, and leave a concise note below.

## Change log

- 2026-07-13: Combined the local and G drive review, froze the primary result, established the execution plan, and documented repository exclusions.
- 2026-07-13: Triaged all visible untracked files. The notebook, scripts, reports, and small diagnostic fixture are intended project artifacts; no files were deleted or staged.
- 2026-07-13: Added `requirements-dev.txt`, installed pytest 8.4.2 in the designated environment, and verified the existing evaluator suite (`3 passed`). The environment emitted a non-blocking `requests` dependency compatibility warning.
- 2026-07-13: Added SFT-record and CLI dry-run regression tests. Implemented `app/app.py` with lazy base/hardmix selection, one active model in memory, latency output, environment configuration, and analytics-disabled dry-run support. Verification: `8 passed`; `app/app.py --dry-run` succeeded without loading model weights.
- 2026-07-13: Added a self-created SVG chart and three built-in Gradio examples. Prepared the adapter model card, Space metadata/dependencies, and release checklist. Rewrote the root README around the frozen full-val result and current demo. Verification: `9 passed`, Gradio dry-run succeeded, and both Hugging Face metadata blocks parsed successfully.
- 2026-07-13: Completed Python syntax checks and prepared `docs/git_staging_plan.md` with three explicit commit groups. Nothing has been staged or committed; P5.2 still requires user authorization for the Git commits.
