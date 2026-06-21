# 推荐路线任务拆分：3B + ChartQA + QLoRA + GPU Space

## 成功标准

项目完成时应具备：

1. baseline 推理脚本；
2. ChartQA 数据转换脚本；
3. 独立评估脚本；
4. QLoRA adapter；
5. base vs fine-tuned 指标对比；
6. Gradio demo；
7. Hugging Face Space；
8. README、实验报告、错误分析和面试材料。

## 阶段 1：项目骨架与环境定义

任务：
- 创建项目目录结构。
- 写依赖清单。
- 写环境检查脚本。
- 写本地、Colab、Space 的职责边界。
- 写国内源优先策略。

目的：
- 防止本地 8GB 显卡被误用来做训练。
- 让后续问题能先归类为依赖、显存、数据、训练或部署问题。

验证：
- `scripts/env_check.py` 可在本地运行并输出 JSON。
- 文档明确本地只做非训练 smoke test。

状态：
- 已开始。

## 阶段 2：baseline 推理

任务：
- 封装 Qwen2.5-VL-3B 加载函数。
- 封装图片 + 问题的 message 构造。
- 支持小样本推理输出 JSONL。
- 记录延迟、显存、生成参数。

目的：
- 建立 base model 基准。
- 提前暴露模型加载、视觉 token、显存问题。

验证：
- Colab 上对 10 条 ChartQA 样本生成答案。
- 本地只在显存允许时做 1 条样本 smoke test。

状态：
- 已完成本地代码骨架和 dry-run。
- 已完成 Colab T4 单图 baseline 验证。
- 已完成 5 条 ChartQA notebook 小样本 baseline 验证。
- 已新增 `scripts/run_chartqa_baseline.py`，待 Colab 用脚本复跑验证。

## 阶段 3：ChartQA 数据转换

任务：
- 加载 `HuggingFaceM4/ChartQA`。
- 转成 Qwen2.5-VL SFT 对话格式。
- 导出 100、1000、full 三档训练文件。
- 保存 split、human_or_machine、question、answer、image metadata。

目的：
- 保证训练数据格式稳定可复现。

验证：
- 随机抽样可视化 20 条，确认图片、问题、答案对应。
- 无空图、空问题、空答案。

状态：
- 未开始。

## 阶段 4：评估脚本

任务：
- 实现答案标准化。
- 实现 exact match、relaxed numeric accuracy、yes/no accuracy。
- 按 human/machine、题型分组统计。
- 导出错误样例。

目的：
- 用指标证明微调效果，而不是只看 loss。

验证：
- 人工构造 10 条预测验证指标逻辑。
- baseline 预测文件可重复评估。

状态：
- 未开始。

## 阶段 5：QLoRA smoke training

任务：
- Colab GPU 上用 100 条样本跑通 QLoRA。
- 保存 adapter。
- 重新加载 adapter 推理。
- 记录 trainable params、显存峰值、loss。

目的：
- 验证训练闭环，而不是一开始就跑 full train。

验证：
- adapter 可重新加载。
- 同一小样本上 base 与 adapter 输出存在变化。

状态：
- 未开始。

## 阶段 6：正式实验

任务：
- 跑 1k 子集。
- 视稳定性扩展到 5k 或 full train。
- 固定最多 2-3 组关键实验。
- 每组保存 config、log、adapter、predictions、metrics。

目的：
- 得到可写入简历的实验结果。

验证：
- 每个结果可追溯到配置和 run id。
- base vs adapter 指标表可复现。

状态：
- 未开始。

## 阶段 7：错误分析

任务：
- 抽样分析错误题。
- 分类 OCR、颜色/图例、算术、问题理解、输出格式错误。
- 只做少量可解释改进实验。

目的：
- 展示模型诊断和工程分析能力。

验证：
- 至少 30 个错误样例有分类和原因。

状态：
- 未开始。

## 阶段 8：Gradio demo

任务：
- 上传图表图片。
- 输入问题。
- 选择 base 或 adapter。
- 输出答案、延迟和可选 prompt。
- 加 3-5 个内置示例。

目的：
- 将模型结果变成可交互产品。

验证：
- 本地或 Colab 能启动 `app.py`。
- adapter 缺失时 demo 不崩溃。

状态：
- 未开始。

## 阶段 9：Hugging Face GPU Space

任务：
- 创建 Gradio Space。
- 上传 app、requirements、README。
- base model 从 HF 加载，adapter 从 model repo 加载。
- 配置 GPU hardware 和 Secrets。

目的：
- 形成公开 demo 链接。

验证：
- Space 构建成功。
- 上传真实图表可返回答案。
- 记录冷启动和推理延迟。

状态：
- 未开始。

## 阶段 10：简历与面试材料

任务：
- README 写清任务、模型、数据、方法、指标、demo。
- 写实验报告和错误分析。
- 写面试问答。

目的：
- 让招聘方快速理解项目价值，也能支撑深入追问。

验证：
- 每个指标都有 run id。
- 面试材料能支持 2 分钟、5 分钟、15 分钟版本讲述。

状态：
- 未开始。
