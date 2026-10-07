# Qwen3-0.6B-Base SFT

本目录是与仓库根目录的手写 TinyStories/FineWeb 模型分开的 Qwen 微调练习。

本地已完成 Qwen3-0.6B-Base 全参数 SFT 和 74 条留出集测试。量化结果与生成失败案例见 [项目结果页](../RESULTS.md)。训练脚本、数据准备脚本和实验清单会提交；原始模型、检查点、优化器状态及未经人工核实的派生问答数据留在本地，不随 Git 发布。

```text
qwen_sft/
├── models/Qwen3-0.6B-Base/       # 本地模型和 tokenizer（Git 忽略）
├── data/
│   ├── legacy_ai_qa/             # 旧 SFT 数据和来源记录
│   │   ├── provenance.jsonl       # 本地生成；每条问题、回答、来源、原划分
│   │   ├── manifest.json          # 旧数据的生成记录
│   │   └── raw/                   # 原始下载文件（Git 忽略）
│   ├── daily_chat.jsonl           # 项目自编日常对话及固定划分
│   ├── train.jsonl                # 本地生成的 Qwen prompt/completion 训练集
│   ├── valid.jsonl                # 本地生成；训练期间验证
│   ├── test.jsonl                 # 本地生成；留到最终固定题对比
│   └── manifest.json              # 输入与输出 SHA-256、长度设置
├── prepare_qwen_sft.py
├── inspect_qwen_sft.py          # 本地单样本 token/label 检查，不训练
└── SFT.md                         # 后续实验步骤
```

在项目根目录运行：

```bash
.venv/bin/python -m pip install -r qwen_sft/requirements.txt
.venv/bin/python qwen_sft/prepare_qwen_sft.py
.venv/bin/python qwen_sft/inspect_qwen_sft.py --prompt hi
.venv/bin/python qwen_sft/train_qwen_sft.py
```

`train_qwen_sft.py` 的实际训练配置写在脚本中的 `SFTConfig`，默认把结果写入被 Git 忽略的 `final_outputs/`。训练前请先核对数据来源、答案、模型许可和设备预算；留出集不会进入训练器。

运行准备脚本前，需要自行取得 Qwen3-0.6B-Base 模型和原始问答数据，并按 `data/legacy_ai_qa/manifest.json` 的来源与哈希核对；然后运行 `data/prepare_ai_qa_sft.py` 生成本地 `provenance.jsonl`。仓库内的 `daily_chat.jsonl` 是自编样本。由于模型权重与派生问答数据未发布，仅克隆仓库不能直接重跑已有模型的评估。

`inspect_qwen_sft.py` 只读取本地 `train.jsonl` 的一条样本和本地 Qwen tokenizer，
调用 TRL 的数据预处理并打印实际 batch 的 token 与 label。它不加载模型权重、
不下载 Hugging Face 示例数据，也不执行训练。默认检查训练集首条；
可用 `--index` 或 `--prompt` 选样本。

脚本默认合并 `data/legacy_ai_qa/provenance.jsonl` 和 `data/daily_chat.jsonl`，
保留各自已有的 `train`、`valid`、`test` 划分，输出三份仅含 `prompt` 和
`completion` 的 JSONL。日常对话是项目自编样本，包含问候、寒暄、感谢、告别、
简单接话和能力说明；其中 48 条进入训练、8 条进入验证、8 条留作测试。
它使用本地 Qwen tokenizer 的聊天模板计算完整问答长度，默认上限 512 token；
可通过 `--max-length` 调整。无效字段、空问答、跨集合重复题、同题冲突答案会报错；
相同集合的完全重复题只保留一条；超长样本被剔除并在 manifest 计数。

训练脚本应只加载 `train.jsonl` 和 `valid.jsonl`。`test.jsonl` 只用于最后的
Base/SFT 同题回答对比。旧数据来自英文 AI/ML 问答，脚本只验证结构和长度，
不证明答案正确或适合目标任务。建议在训练前人工核查实际要使用的来源与答案，
尤其是涉及会变化的产品/API 描述。

新增样本让短跑训练覆盖常见寒暄，但不能保证 Base 模型凭几十条样本就能
稳定聊天。最终仍要用 `test.jsonl` 中未见过的问候和接话题比较 Base 与 SFT。

如果要从旧原始数据重建来源记录，先运行
`data/prepare_ai_qa_sft.py`；它仍使用根目录手写模型的 tokenizer 和检查点，
输出到 `qwen_sft/data/legacy_ai_qa/`，随后再运行本目录脚本。

## 训练后测试

下面的脚本只读取未参与训练和选模的 `test.jsonl`。运行量化测试后，
`final_outputs/test_metrics.json` 会保存测试文件校验值、completion loss、困惑度和逐 token 命中率：

```bash
.venv/bin/python qwen_sft/test_qwen_sft.py --include-base
```

逐 token 命中率是在给定标准答案前文时预测下一个 token 的比例，**不是回答事实正确率**。
开放式问题请看固定题生成对比。默认取测试集前 10 题；`--limit 0` 会生成全部 74 题。
两边使用相同的模板、输入与贪心解码参数，输出 Markdown 表格与原始 JSON：
生成时设置 `enable_thinking=False`，与训练样本中空的 `<think>…</think>` 段保持一致。
两边默认使用相同的 `repetition_penalty=1.1` 抑制重复生成。

```bash
.venv/bin/python qwen_sft/compare_qwen_sft.py
```

最后进入交互模式。默认每题独立，`--multi-turn` 才保留上下文；
每次最多生成 64 个新 token（可用 `--max-new-tokens` 调整）；
输入 `/reset` 清空上下文，`/exit` 退出：

```bash
.venv/bin/python qwen_sft/generate_qwen_sft.py
```
