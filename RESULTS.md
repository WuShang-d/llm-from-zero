# 实验结果与边界

本页汇总已运行的实验。仓库保存代码、分词器文件与精简记录；语料、模型权重、优化器状态和未经人工核实的派生问答数据留在本地。下列数值来自各次已有实验，**不同模型、数据和词表的 loss 不能横向比较**。

| 实验 | 配置与数据 | 可核实的结果 | 主要观察 |
| --- | --- | --- | --- |
| TinyStories 预训练 | 手写 BPE、16,913,280 参数、256 token 上下文 | 已完成训练和验证链路；[代码](train.py)与 [训练记录](CHAT_LEARNING_RECORD.md) | 适合验证模型实现，不能据此声称通用对话能力 |
| 17M 对话 SFT | 教学用的 28 个训练样本、5 个验证样本，100 步 | 验证 loss 在第 50 步最低，为 2.0869；第 100 步为 2.3194 | 训练 batch loss 接近零后验证 loss 回升；生成仍会答非所问 |
| FineWeb-Edu 预训练 | 97,536,768 参数、512 token 上下文、一轮预训练 | 最佳检查点记录的验证 loss 约 2.7676 | 这是续写模型的验证指标，不是问答准确率 |
| 100M 领域 SFT | 528 个训练、66 个验证样本，300 步 | 最佳检查点为第 275 步，验证 loss 2.2217 | 实测对话会重复文本、混淆概念；loss 下降没有解决回答质量 |
| Qwen3-0.6B-Base 全参数 SFT | 576 个训练、74 个验证、74 个留出测试样本；训练 2 epoch | 同一留出集上的 completion loss：Base 3.1477，SFT 2.0026；逐 token 命中率：48.1% → 59.0% | 参考答案预测能力改善；固定题生成仍有乱码、重复和事实错误 |

17M 与 100M 的训练细节和原始失败案例在 [学习记录](CHAT_LEARNING_RECORD.md)。Qwen 的数据准备、训练和评估命令在 [Qwen README](qwen_sft/README.md)。Qwen 指标来自本地 `final_outputs/test_metrics.json`：留出集 SHA-256 为 `2489b329dd95431ddfafbf51f5a9fd30664f71e4103a5d23d42ec56317f8c08e`，使用 teacher forcing 计算 completion 指标；逐 token 命中率**不是**事实正确率。训练数据仅完成结构与长度检查，答案仍需人工核实。

## 固定题生成例子

以下摘录自 Qwen 留出集前 10 题的 Base/SFT 同题记录。两边使用相同的 tokenizer、聊天模板、greedy 解码、`max_new_tokens=128` 与 `repetition_penalty=1.1`。摘录用省略号表示截断，不能作为完整回答；结论以原始本地记录为准。

| 测试题 | Base 开头 | SFT 开头 | 观察 |
| --- | --- | --- | --- |
| How does streaming work in LangGraph agents? | `How does streaming work in LangGraph agents? ⚐� ⚐ ⚐ ...` | `الإلك ... LangGraph agents use a streaming model ...` | SFT 开始形成句子，但仍有异常字符且内容泛泛 |
| What is AutoGen Studio and what are its advantages? | `What is AutoGen Studio and what are its advantages? ⚐� ...` | `الإلك ... AutoGen Studio is a powerful AI tool designed to generate high-quality content ...` | SFT 的回答偏离问题，不能据此认定掌握概念 |

17M 的一个多轮失败案例：问 `what is ai`，模型答 `Aiquorms can fly.`；问 `Tell me about a tree.`，模型答 `Could I take a picture of a cat?`。完整上下文和说明见 [学习记录](CHAT_LEARNING_RECORD.md)。这些例子说明项目目前的价值在于实现、实验设计和错误分析，而不在模型实用性能。

## 复现与检查

1. 安装 [根目录依赖](requirements.txt)，运行 `python -m unittest discover -s model -p 'test_*.py'` 和 `python -m unittest discover -s tokenizer -p 'test_*.py'`。
2. 手写模型按 [README](README.md) 准备 TinyStories 或 FineWeb-Edu 数据后运行 `train.py`；训练产物默认写入被 Git 忽略的 `checkpoints/`。
3. Qwen 实验按 [Qwen README](qwen_sft/README.md) 准备官方 Base 模型与数据，先检查训练标签，再训练并对同一留出集运行测试。`qwen_sft/final_outputs/` 和 `qwen_sft/outputs/` 只保存本地大文件，不提交。

仓库没有发布模型权重，所以无法仅凭克隆仓库重放上述已训练模型的生成输出；可以复现代码测试与从头训练流程。已有实验记录也缺少一份统一的运行环境快照，跨设备复跑数值可能不同。
