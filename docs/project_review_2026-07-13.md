# ChartQA project review - 2026-07-13

## Scope

This review combines two sources:

- local repository: `C:\Users\90553\Desktop\Wireless Simulation\LLM_project`
- experiment archive: `G:\我的云端硬盘\qwen25vl-chartqa-qlora`

The local repository is the source of truth for code and documentation. The G drive is currently the source of truth for large datasets, adapters, predictions, and raw experiment outputs. No G drive files were modified during the review.

## Original goal

The project was defined as a resume-ready end-to-end ChartQA system based on Qwen2.5-VL-3B and QLoRA. Its intended deliverables are baseline inference, data preparation, low-memory training, reproducible evaluation, a Gradio application, Hugging Face Space deployment, and an interview-ready experiment narrative.

The research portion is substantially complete. The application, deployment, repository cleanup, and public reproduction path are not complete.

## Verified final results

All seven full ChartQA validation runs are present on the G drive. Each run contains 1,920 evaluated samples.

| Run | Exact | Relaxed | Relaxed delta vs baseline |
|---|---:|---:|---:|
| `baseline_default` | 65.73% | 75.94% | 0.00 pp |
| `standard_steps100` | 68.59% | 77.24% | +1.30 pp |
| `standard_numeric_final` | 68.23% | 76.98% | +1.04 pp |
| `experiment_a_steps200` | 69.01% | 77.55% | +1.61 pp |
| `experiment_b_calcnum` | 68.91% | 77.45% | +1.51 pp |
| `experiment_d_hardmix` | 69.32% | **77.86%** | **+1.93 pp** |
| `experiment_f_steps250_r16a32` | **69.48%** | 77.66% | +1.72 pp |

The deployment candidate is `experiment_d_hardmix`. It has the best relaxed accuracy and uses a 74.4 MB adapter. `experiment_f_steps250_r16a32` remains the exact-accuracy comparison, but its 148.7 MB adapter is not the preferred deployment artifact.

The best hardmix improvement is concentrated on human-authored questions:

| Split | Baseline relaxed | Hardmix relaxed | Delta |
|---|---:|---:|---:|
| Human | 68.96% | 72.19% | +3.23 pp |
| Machine | 82.92% | 83.54% | +0.63 pp |

The seven-run relaxed oracle is 83.07% (1,595/1,920), leaving 325 all-runs-wrong samples. This is a diagnostic upper bound, not a deployable result.

## Secondary findings

- The fixed random val100 benchmark agrees with the full validation trend, but is too small to rank close adapters reliably.
- The trained adapters occupy a narrow 0.62 pp relaxed range. More small LoRA hyperparameter ablations have low expected value without multiple seeds.
- Tested 7B configurations reached 72%-75% relaxed on val100 and did not beat the comparable 3B result of 76%.
- Increasing `max_pixels` did not resolve the selected hard failures.
- Staged extraction recovered 15/77 and 16/77 hard samples in its two modes; their oracle recovered 23/77. It is useful research evidence but not yet a production pipeline.
- Evaluator cleanup found reference, list-format, scale, color, and semantic-equivalence issues. Normalization v2 recovered four additional diagnostic cases, but it was designed after failure inspection and must remain a secondary diagnostic metric.
- Targeted prompt routing recovered only 1/28 true-hard samples. More prompt variants are not the recommended next step.

## Artifact status

Six complete adapters are present on the G drive, including their weight, PEFT configuration, processor, and tokenizer files. The full-val predictions and metrics are also present. The local 24A script and notebook module exist, but no corresponding 24A result directory was found on the G drive; 24A must therefore be treated as planned, not completed.

## Engineering gaps

1. `app/app.py` does not exist, so the Gradio and Space milestones remain incomplete.
2. Many current scripts and experiment documents are untracked. A fresh clone does not represent the actual project state.
3. There is no single run registry connecting configuration, data slice, evaluator version, metrics, and artifact location.
4. Existing evaluator tests use pytest conventions, but pytest is not installed in the checked local environment. `unittest discover` ran zero tests, so there is no verified passing test suite yet.
5. The G drive is a private archive, not a public artifact distribution mechanism. Final adapters should be published to a versioned Hugging Face repository.

## Direction decision

Freeze the main research result now. Do not start another prompt or small adapter ablation before the engineering milestones are complete. The recommended sequence is:

1. freeze the run registry and evaluation policy;
2. make tests executable;
3. organize and commit the current code and selected reports;
4. implement the minimum Gradio application;
5. publish the hardmix adapter and model card;
6. deploy and verify the Hugging Face Space;
7. rewrite the README around the reproducible final result.

Current estimated completion by area:

| Area | Completion |
|---|---:|
| Data and training | 90% |
| Evaluation | 90% |
| Error analysis | 95% |
| Experiment management | 55% |
| Automated tests | 30% |
| Demo | 10% |
| Deployment | 0% |
| Portfolio presentation | 50% |
