# Qwen2.5-VL-3B + ChartQA + QLoRA + Gradio + Hugging Face Space 项目规划

## 0. 项目定位

这是一个面向简历和 AI 工程师面试的端到端视觉语言模型项目。最终产物不是只跑通一个 notebook，而是形成一条能展示工程能力的闭环：

1. 用 Qwen2.5-VL-3B-Instruct 做图表问答 baseline。
2. 用 ChartQA 构造多模态监督微调数据。
3. 用 QLoRA/LoRA 完成低成本领域适配。
4. 用一致的评估脚本比较 base model、微调 adapter、可选 prompt baseline。
5. 用 Gradio 做可交互 demo，并部署到 Hugging Face Space。
6. 用 README、模型卡、实验表、失败案例和面试讲稿说明技术决策。

核心面试叙事：我把一个通用 VLM 适配到图表理解场景，解决了数据格式、显存、训练稳定性、评估复现、线上部署和用户演示的问题。

## 1. 规划假设

- 模型：`Qwen/Qwen2.5-VL-3B-Instruct`，不从零训练视觉编码器。
- 数据：优先使用 `HuggingFaceM4/ChartQA`。该数据集在 HF 上约 32.7k 条，包含 train/val/test、image、query、label、human_or_machine 字段，适合作为可复现实战数据。
- 训练：优先用 Colab L4/A100/H100 跑 QLoRA 或 LoRA。你本地 8GB 显卡只承担小样本推理、数据格式检查和 UI smoke test。
- 部署：Hugging Face Space 用 Gradio。免费 CPU 不作为正式部署目标；3B VLM 更现实的部署目标是 T4/L4/A10G 级别 GPU Space，或用外部推理 API 作为备选。
- 协作：本地 Codex 负责代码结构、文档、debug 方案；Colab 负责 GPU 执行。GitHub 仓库作为同步主线，Google Drive 只放大文件缓存和 checkpoint 备份。

## 2. 可行性依据

- Qwen2.5-VL-3B 官方模型卡说明它支持文本、图像、图表、布局和结构化输出场景，并提供 Transformers 推理示例、`qwen-vl-utils` 用法、`min_pixels/max_pixels` 控制视觉 token 成本的方式。参考：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL 已被 Transformers 文档支持，可用官方 processor/model 类加载。参考：https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl
- ChartQA HF 数据集已整理成 parquet，约 964MB，train/val/test 划分明确，可以直接用 `datasets.load_dataset` 拉取。参考：https://huggingface.co/datasets/HuggingFaceM4/ChartQA
- TRL 的 SFTTrainer 支持 VLM SFT，并支持 PEFT adapter 训练；assistant-only loss、completion-only loss、packing 等能力可用于控制训练目标。参考：https://huggingface.co/docs/trl/en/sft_trainer
- PEFT + bitsandbytes 官方文档确认 4-bit + LoRA 是标准 QLoRA 路径，NF4、double quant、bf16 compute 是常用配置。参考：https://huggingface.co/docs/peft/en/developer_guides/quantization 和 https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- Hugging Face Spaces 是 Git 仓库，Gradio Space 会根据 `README.md` YAML、`app.py`、`requirements.txt` 自动构建；Space 默认 CPU 资源有限，但可升级 GPU。参考：https://huggingface.co/docs/hub/en/spaces-overview 和 https://huggingface.co/docs/hub/en/spaces-sdks-gradio
- Colab 官方 FAQ 确认 notebook 可从 Drive 或 GitHub 加载，GPU 类型和资源会动态变化，付费用户可获得更多算力但仍需考虑会话与资源波动；Colab 也提供内置 AI 代码辅助。参考：https://research.google.com/colaboratory/faq.html
- Qwen 官方 GitHub 提供了 Qwen2.5-VL/Qwen3-VL 微调脚本、数据格式、LoRA 参数、`max_pixels/min_pixels` 等训练实践参考。参考：https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- 社区中已有 ChartQA + Qwen2-VL + TRL + Space 的尝试，但能看到 7B 部署因 `device_map/offload_dir` 等问题报错。这说明项目方向可行，但部署必须尽早验证，且 3B 比 7B 更适合作为简历项目主线。参考：https://huggingface.co/spaces/sergiopaniego/Qwen2-VL-7B-trl-sft-ChartQA

## 3. 三种路线对比

### 路线 A：推荐路线，3B + ChartQA + QLoRA + GPU Space

做法：用 Qwen2.5-VL-3B-Instruct 做 baseline，用 ChartQA 做 SFT，QLoRA 训练 adapter，Gradio 加载 base model + adapter，部署到 T4/L4/A10G Space。

优点：最符合简历项目目标；覆盖数据、训练、评估、部署；显存压力可控；能讲清工程闭环。

风险：VLM QLoRA 数据 collator 和图像预处理容易踩坑；Space 冷启动慢；依赖版本要固定。

可行性验证：先用 100 条样本完成单步训练和 adapter 保存，再扩大到 1k/5k/full；Space 先加载 base 3B 4-bit 推理，再接 adapter。

### 路线 B：轻量路线，prompt baseline + LoRA 小样本 + 本地/Colab demo

做法：不追求全量训练，只做 500-2000 条样本微调，重点展示系统完整性和误差分析。

优点：最快得到可展示结果；适合先构建项目骨架；8GB 本地也能做部分 smoke test。

风险：指标提升可能有限；面试中如果被问训练规模，需要坦诚说明它是小样本验证。

可行性验证：base 与 adapter 在 val 子集上至少能跑出稳定差异，并形成错误类别分析。

### 路线 C：高配路线，3B/7B 对比 + 多数据集增强 + 更强部署

做法：在 ChartQA 外加入 PlotQA、FigureQA、DocVQA/chart subset 或自制中文图表 QA，比较 3B 与 7B。

优点：技术深度更强，适合冲击更高级面试。

风险：范围大，容易变成堆实验；7B 部署和显存成本更高；数据许可证与格式差异会增加复杂度。

可行性验证：只有路线 A 完整跑通后再扩展，否则不进入主线。

推荐：先做路线 A，保留路线 B 的小样本快速闭环作为第一里程碑，路线 C 作为加分项。

## 4. 学习方向

### 4.1 多模态模型基础

学习内容：VLM 输入格式、image token、chat template、processor、视觉 token 预算、`min_pixels/max_pixels` 对显存和精度的影响。

学习目的：知道为什么同一张图改变分辨率会影响训练显存、推理速度和图表细节理解。

验证方式：用同一张 ChartQA 图片，在不同 `max_pixels` 下比较显存、延迟和答案变化。

### 4.2 Qwen2.5-VL 工程用法

学习内容：`Qwen2_5_VLForConditionalGeneration`、`AutoProcessor`、`qwen_vl_utils.process_vision_info`、消息格式、本地图片/URL/base64 输入。

学习目的：把模型调用封装成可测试函数，避免 notebook 里散落不可复用代码。

验证方式：准备 3 张图表样例，脚本能稳定返回答案，并记录输入 token、输出 token、延迟。

### 4.3 ChartQA 数据与评估

学习内容：数据字段、human/machine 问题差异、答案标准化、exact match、relaxed accuracy、数值容差、yes/no 分类。

学习目的：避免只看训练 loss；用任务指标证明微调有效。

验证方式：在 val/test 子集上输出 `metrics.json`，至少包含总体准确率、human/machine 分组准确率、数值题/文本题/yes-no 题分组。

### 4.4 QLoRA/LoRA 微调

学习内容：4-bit NF4、double quant、LoRA rank/alpha/dropout、target modules、gradient accumulation、bf16/fp16、paged optimizer。

学习目的：能解释为什么不全量微调，为什么 3B + QLoRA 是工程上合理的选择。

验证方式：训练日志中必须能看到 trainable params、显存峰值、loss 下降、checkpoint/adapters 可重新加载。

### 4.5 Gradio 与 Hugging Face Space 部署

学习内容：`app.py`、`requirements.txt`、Space YAML、Secrets、GPU 选择、冷启动、模型缓存、adapter 加载。

学习目的：把训练结果变成用户可试的 demo，而不是停留在 notebook。

验证方式：Space 上能上传一张图、输入问题、返回答案；README 有示例图、示例问题和已知限制。

### 4.6 工程复现与面试表达

学习内容：实验配置版本锁定、随机种子、数据子集采样、日志、模型卡、失败案例、ablation。

学习目的：面试时能讲“为什么这么做”和“怎么证明有效”，而不是只说“我微调了模型”。

验证方式：别人按 README 能复现 baseline evaluation、小样本训练和 Space 本地启动。

## 5. 实现路线

### 模块 1：项目骨架与环境定义

做什么：
- 建立 `src/`、`scripts/`、`notebooks/`、`configs/`、`app/`、`docs/`、`outputs/` 结构。
- 固定依赖：`transformers`、`accelerate`、`datasets`、`peft`、`trl`、`bitsandbytes`、`qwen-vl-utils`、`gradio`、`evaluate`、`wandb` 或 `tensorboard`。
- 写 `env_check.py`：打印 Python、CUDA、torch、transformers、GPU 名称、显存、bitsandbytes 可用性。

目的：让本地、Colab、Space 三个环境的差异可见。

可行性验证：
- 本地用指定 Python 环境运行环境检查。
- Colab 运行同一脚本能识别 T4/L4/A100/H100。
- 如果本地 Windows 下 bitsandbytes 不稳定，本地只做非训练 smoke test，训练转 Colab。

交付物：
- `requirements.txt`
- `scripts/env_check.py`
- `docs/environment.md`

### 模块 2：baseline 推理

做什么：
- 封装 `load_model()`、`build_messages()`、`predict_chartqa()`。
- 支持本地图片路径和 PIL image。
- 默认低视觉 token 配置：先用较小 `max_pixels` 保证能跑，再逐步提高。
- 输出答案、延迟、显存、生成参数。

目的：先证明 base model 可以完成 ChartQA 任务，建立后续微调对比基准。

可行性验证：
- 从 ChartQA 取 10 条样本，base model 能完成推理。
- 本地 8GB 如果 4-bit 仍 OOM，则在 Colab 跑；本地保留小模型或 CPU smoke test。
- 每次推理结果写入 JSONL，避免只看屏幕输出。

交付物：
- `src/infer.py`
- `scripts/run_baseline_samples.py`
- `outputs/baseline_samples.jsonl`

### 模块 3：ChartQA 数据转换

做什么：
- 用 `datasets.load_dataset("HuggingFaceM4/ChartQA")` 加载数据。
- 将每条数据转换成 Qwen VLM 对话格式：
  - user：图片 + 问题，例如 `Answer the chart question. Give a concise answer.`
  - assistant：标准答案，取 `label[0]`。
- 保留字段：sample id、split、human_or_machine、原问题、标准答案、图片尺寸。
- 做小样本导出：100、1000、full 三档。

目的：把公开数据集变成稳定、可训练、可评估的数据格式。

可行性验证：
- 随机抽 20 条渲染检查图片、问题、答案是否对应。
- 数据转换后没有空图、空问题、空答案。
- 训练格式中的图片数量与 `<image>` 或 message image 数量严格一致。

交付物：
- `src/data_chartqa.py`
- `scripts/prepare_chartqa.py`
- `data/processed/chartqa_train_100.jsonl`
- `data/processed/chartqa_val.jsonl`

### 模块 4：评估脚本

做什么：
- 写独立评估脚本，不依赖训练代码。
- 实现答案标准化：小写、去空格、去标点、百分号/小数兼容。
- 实现指标：
  - exact match
  - relaxed numeric accuracy
  - yes/no accuracy
  - human vs machine 分组
  - 错误样例导出

目的：让微调收益可量化，面试时能展示模型在哪些题型上提升或失败。

可行性验证：
- 用人工构造的 10 条预测/答案单元测试指标逻辑。
- base model 在固定 val 子集上能产出 `metrics.json`。
- 同一预测文件重复评估结果一致。

交付物：
- `src/eval_chartqa.py`
- `scripts/evaluate_predictions.py`
- `outputs/baseline_metrics.json`

### 模块 5：QLoRA 小样本训练闭环

做什么：
- 先用 100 条样本跑通完整训练：加载 4-bit base model，挂 LoRA adapter，训练若干 step，保存 adapter。
- 使用 PEFT 配置：`r=8/16`、`lora_alpha=16/32`、`lora_dropout=0.05` 起步。
- 训练目标优先只算 assistant answer loss。
- 视觉 token 先控制在较低上限，确认稳定后提高。

目的：先验证“训练代码、数据 collator、adapter 保存/加载、评估”全部闭环。

可行性验证：
- 单卡 Colab 能跑完 100 条样本训练。
- `trainer.train()` 不报图像 batch/collator 错。
- adapter 保存后重新加载，推理输出发生可观变化。
- 训练日志记录显存峰值、loss、step time。

交付物：
- `src/train_qlora.py`
- `configs/train_qlora_smoke.yaml`
- `outputs/adapters/smoke/`
- `outputs/smoke_train_log.md`

### 模块 6：正式微调实验

做什么：
- 从 1k 子集开始，再扩展到 full train。
- 固定 2-3 组关键实验，不做无边界调参：
  - base prompt baseline
  - QLoRA r=8
  - QLoRA r=16 或更高视觉分辨率
- 使用 Colab 大显存 GPU；如果 A100/H100 可用，优先提高 batch/eval throughput，而不是无节制扩大实验范围。

目的：得到能写进简历的量化结果和可解释实验。

可行性验证：
- 每组实验都有 `config.yaml`、训练日志、adapter、预测文件、metrics。
- 至少在 val/test 子集上比较 base 与 adapter。
- 若 full train 成本或稳定性不佳，保留 1k/5k 实验作为主结果，并写清限制。

交付物：
- `configs/experiments/*.yaml`
- `outputs/runs/<run_id>/metrics.json`
- `outputs/runs/<run_id>/predictions.jsonl`
- `docs/experiment_report.md`

### 模块 7：错误分析与加分项

做什么：
- 抽样分析错误题，按类别归因：
  - OCR 读数错误
  - 图例/颜色混淆
  - 算术错误
  - 问题理解错误
  - 输出格式不规范
- 加入可选改进：
  - prompt 约束简短答案
  - numeric postprocess
  - 提高 `max_pixels`
  - 加入少量中文图表 QA 自制数据

目的：展示你不是只会跑训练，而是会诊断模型行为。

可行性验证：
- 错误分析表至少包含 30-50 个失败样例。
- 每个改进只做一个变量变化，避免无法解释。
- 加分项不影响主线可复现。

交付物：
- `docs/error_analysis.md`
- `outputs/error_cases.csv`
- 可选 `data/custom_zh_chartqa/`

### 模块 8：Gradio 本地 Demo

做什么：
- 做一个简洁但面试友好的 UI：
  - 上传 chart image
  - 输入 question
  - 选择 base / fine-tuned adapter
  - 输出 answer、latency、可选 prompt
  - 提供 3-5 个内置 examples
- 本地 Gradio 先用 Colab 或本地低配模式验证。

目的：把模型能力变成可交互产品，方便面试展示和简历链接。

可行性验证：
- `python app/app.py` 能在本地或 Colab 启动。
- examples 不依赖本地绝对路径。
- adapter 缺失时能清楚提示，而不是崩溃。

交付物：
- `app/app.py`
- `app/examples/`
- `app/README.md`

### 模块 9：Hugging Face Space 部署

做什么：
- 新建 Space，SDK 选 Gradio。
- Space 仓库包含 `app.py`、`requirements.txt`、`README.md` YAML。
- 模型权重策略：
  - base model 从 HF 加载；
  - adapter 上传到 HF model repo；
  - Space 启动时加载 base + adapter；
  - 如果冷启动太慢，改成 merged checkpoint 或更强 GPU。
- Secrets 放 `HF_TOKEN`，不要硬编码。

目的：产出公开可访问 demo 链接，形成简历亮点。

可行性验证：
- Space 首次构建成功。
- 上传真实 chart 后能返回答案。
- 记录冷启动时间和单次推理延迟。
- 如果 GPU Space 仍 OOM，降低 `max_pixels`、改 4-bit 加载、或使用更高显存 GPU。

交付物：
- Hugging Face Space URL
- Hugging Face adapter/model repo URL
- `docs/deployment.md`

### 模块 10：Colab、GitHub、Google Drive 同步流程

做什么：
- GitHub 存代码、notebook、配置、文档。
- Google Drive 存临时 checkpoint、下载后的 ChartQA cache、训练日志备份。
- Colab 每次启动：
  1. clone/pull GitHub repo
  2. install requirements
  3. mount Drive
  4. link cache/checkpoints
  5. run env check
  6. run selected script
- 本地 Codex 修改代码后 push；Colab 只执行和记录，不作为主编辑环境。

目的：避免 Colab 临时文件丢失，也避免 Drive 成为代码主版本。

可行性验证：
- 新 Colab runtime 从零启动能在一套 notebook cell 内复现 baseline。
- Drive checkpoint 能被重新加载。
- GitHub commit hash 写入每次实验日志。

交付物：
- `notebooks/colab_train.ipynb`
- `notebooks/colab_eval_and_demo.ipynb`
- `docs/colab_workflow.md`

### 模块 11：在 Colab 中使用 AI 工具辅助调试

推荐做法：
- 主力仍用本地 Codex + GitHub 管理代码。
- Colab 内置 AI/Gemini 用来解释报错、生成临时 notebook cell、修小片段。
- 对复杂 bug，把完整 traceback、当前 commit、config、最小复现样本带回本地 Codex 修。

谨慎做法：
- 不建议把 Codex CLI 长期跑在 Colab 里当远程开发环境，因为 Colab runtime 临时、网络/权限/secret 管理不稳定。
- 如果一定要试，只用于短时诊断，并把所有改动通过 git diff 回传，不在 Colab 里产生不可追踪代码。

可行性验证：
- 任意一次 Colab AI 或 Codex 生成的修改，必须能在 GitHub diff 中审查。
- 不把 HF token、WandB key、Google Drive 私密路径写入 notebook 输出。

交付物：
- `docs/debug_workflow.md`
- `docs/common_errors.md`

### 模块 12：简历与面试材料

做什么：
- README 首屏写清：任务、模型、数据、方法、指标、Demo 链接。
- 增加架构图：数据准备 -> baseline -> QLoRA -> eval -> Gradio -> Space。
- 增加实验表：base vs fine-tuned，human/machine 分组。
- 增加面试问答：
  - 为什么选 Qwen2.5-VL-3B？
  - 为什么 QLoRA 而不是 full fine-tune？
  - ChartQA 指标怎么设计？
  - 显存不够怎么处理？
  - 部署时遇到什么问题？
  - 这个项目的局限是什么？

目的：让项目能被招聘方快速理解，也能支撑深入追问。

可行性验证：
- README 里每个结果都能追溯到 run id。
- Demo 链接可打开。
- 面试讲稿能在 2 分钟、5 分钟、15 分钟三个粒度讲清楚。

交付物：
- `README.md`
- `docs/interview_notes.md`
- `docs/project_retrospective.md`

## 6. 最小成功标准

项目达到以下标准，就可以作为简历项目投递：

1. 有可运行的 baseline 推理脚本。
2. 有 ChartQA 数据转换脚本。
3. 有独立评估脚本和至少一份 baseline metrics。
4. 有一个 QLoRA adapter，可重新加载推理。
5. 有 base vs fine-tuned 的指标对比。
6. 有 Gradio demo，可本地启动。
7. 有 Hugging Face Space 或等价公开 demo。
8. README 能让陌生人复现核心流程。
9. 有错误分析，说明模型还不会什么。
10. 有清楚的面试叙事和技术取舍。

## 7. 推荐执行顺序

1. 环境检查与依赖锁定。
2. 单张图 baseline 推理。
3. ChartQA 10 条样本推理。
4. ChartQA 数据转换。
5. baseline evaluation 子集跑通。
6. 100 条 QLoRA smoke training。
7. adapter 重新加载并评估。
8. 1k/5k/full 正式实验。
9. 错误分析和 prompt/分辨率小实验。
10. Gradio 本地 demo。
11. Space 部署。
12. README、实验报告、面试材料整理。

## 8. 主要风险与备选方案

风险：本地 8GB 显存不足。

备选：本地只做数据、UI、CPU/小样本 smoke test；训练与正式评估放 Colab GPU。

风险：bitsandbytes 在 Windows 环境不稳定。

备选：本地不训练；Colab Linux 跑 QLoRA；本地用已训练 adapter 做轻量测试。

风险：Space 冷启动慢或 OOM。

备选：降低 `max_pixels`，使用 4-bit，加 GPU Space，或把 adapter merge 后部署；如果仍不稳，用 Space 调外部推理端点。

风险：微调指标没有明显提升。

备选：检查答案标准化、训练 loss mask、数据格式；优先在人类问题/机器问题、数值题/文本题分组看局部提升；把失败分析作为项目深度的一部分。

风险：项目范围膨胀。

备选：路线 A 是主线；7B、多数据集、中文扩展只作为加分项，不阻塞主线交付。

## 9. 自查清单

- 没有按天排期，只有按模块的顺序拆分。
- 每个模块都有目的和可行性验证。
- 主线聚焦 Qwen2.5-VL-3B + ChartQA + QLoRA + Gradio + Space。
- 本地 8GB、Colab 大显存、GitHub、Google Drive、Colab AI/Codex 调试方式都已纳入。
- 规划服务于简历和 AI 工程师面试，而不是单纯复现教程。
- 没有把 7B、多数据集、复杂部署放进必须完成路径。

