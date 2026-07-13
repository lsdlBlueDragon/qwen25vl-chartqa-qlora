# Project execution plan

This is the authoritative implementation plan. Status values are `completed`, `in progress`, `pending`, and `blocked`.

## Success criteria

The project is complete when a fresh clone can install documented dependencies, run evaluator tests, reproduce evaluation from a prediction fixture, start a Gradio demo with a configured base model and adapter, and reach a deployed Hugging Face Space. Published metrics must identify their dataset slice and evaluator policy.

## Plan and completion status

| ID | Work item | Verification | Status |
|---|---|---|---|
| P0.1 | Review local repository and G drive archive | Review records both sources and reconciles final metrics | completed |
| P0.2 | Freeze final model and metric policy | `configs/final_runs.yaml` identifies deployment and comparison runs | completed |
| P0.3 | Stop low-value prompt ablations | 24A is recorded as planned, not completed, and is not on the critical path | completed |
| P1.1 | Establish documentation entry points | Review, plan, handoff, and Git exclusion policy exist | completed |
| P1.2 | Define repository inclusion/exclusion policy | `.gitignore` and policy document agree | completed |
| P1.3 | Triage all untracked files | Every untracked file is intentionally included, ignored, archived, or removed with approval | completed |
| P1.4 | Move safety backups off the C drive | Verified recovery archive and Git bundle exist on F or E; C contains no backup files | completed |
| P2.1 | Make pytest tests executable | `python -m pytest -q` runs and passes non-GPU tests | completed |
| P2.2 | Add focused data and CLI tests | SFT record and dry-run behavior have regression coverage | completed |
| P3.1 | Implement `app/app.py` | Local/Colab Gradio smoke test accepts image and question | completed |
| P3.2 | Support base and hardmix modes | Both modes load through documented configuration | completed |
| P3.3 | Add examples and latency output | Demo workflow is complete without notebook steps | completed |
| P4.0 | Prepare publishable adapter and Space packages | Model card and Space packaging files validate locally | completed |
| P4.1 | Publish hardmix adapter | Versioned Hugging Face adapter repository and model card are available | blocked |
| P4.2 | Deploy GPU Space | Public Space passes upload/question inference smoke test | blocked |
| P5.1 | Rewrite README | README contains architecture, final table, reproduction commands, demo link, and limitations | completed |
| P5.2 | Final clean-clone verification | Documented setup and non-GPU tests pass from a clean checkout | completed |

P4 is marked blocked because publishing and deployment require the user's Hugging Face account, repository choices, credentials, and potentially paid GPU hardware. Local preparation for P4 can proceed without those credentials.

## Frozen decisions

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Deployment adapter: `chartqa_qlora_hardmix1k_steps100`
- Exact comparison adapter: `chartqa_qlora_train1k_steps250_r16a32`
- Primary benchmark: complete ChartQA validation split, 1,920 samples
- Primary model-selection metric: existing frozen relaxed accuracy
- Secondary metrics: exact, numeric relaxed, human relaxed, machine relaxed
- Normalization v2: diagnostic only
- 7B, higher-resolution, prompt-routing, and 24A work: outside the current critical path

## Immediate next execution block

1. Resume P4.1/P4.2 after the user supplies Hugging Face repository decisions, credentials, and GPU hardware choice.

The plan should be updated after every completed block. Do not create another dated plan unless this file is explicitly retired.
