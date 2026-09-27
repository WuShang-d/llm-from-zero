# 从续写模型到对话模型：四课实验教案

这套实验先复用现有的约 17M 参数 TinyStories 检查点和 8192 词表，把对话训练链路跑通。自带的 24 条训练对话、4 条验证对话是**教学样例**，只用于检查格式、损失和生成流程；反复训练会记住样例，不能据此判断通用对话能力。不会改写原来的 `train.py`、`generate.py` 或 `checkpoints/best_model.pt`。

## 第 1 课：数据怎样变成模型看到的 token

**目标**：理解 `User:`、`Assistant:`、`<|bos|>`、`<|eos|>` 的作用，以及下一 token 预测中的输入与标签错位。

```bash
cd ~/Documents/llm-from-zero
python chat_sft.py inspect
```

阅读 `data/chat_demo_train.jsonl` 和 `chat_data.py`。自己新增一条两轮对话，再运行 `inspect`。思考：为什么第二轮要保留第一轮作为上下文？

**完成标准**：能指出哪些 token 作为条件输入，哪些 token 被计入损失。

## 第 2 课：只训练助手回答

**目标**：理解监督微调（SFT）、`ignore_index=-100`、训练集和验证集的区别。

先做不保存权重的链路检查，再运行教学训练：

```bash
python chat_sft.py train --steps 2 --eval-every 1 --no-save
python chat_sft.py train --steps 40 --eval-every 10
```

结果单独存为 `checkpoints/chat_demo.pt`。记录训练损失与验证损失。试着把 `--steps` 增大，观察训练损失继续下降时验证损失是否也下降。

**完成标准**：能解释为什么只对助手回答及结束标记计算损失，以及为何小样本容易过拟合。

## 第 3 课：生成与人工评估

**目标**：区分“程序能生成文字”和“模型真的会对话”。

```bash
python chat_sft.py chat --prompt "Hello!" --temperature 0
python chat_sft.py chat --prompt "What can I do on a rainy day?" --temperature 0
python chat_sft.py chat
```

用至少五个**没有出现在训练集**的问题比较原始检查点与微调检查点：

```bash
python chat_sft.py chat --checkpoint checkpoints/best_model.pt --prompt "Tell me about a tree." --temperature 0
python chat_sft.py chat --checkpoint checkpoints/chat_demo.pt --prompt "Tell me about a tree." --temperature 0
```

记录是否遵循问答格式、是否结束回答、是否答非所问。不要只看单个提示词或单次随机采样。

**完成标准**：写出至少两个改进和两个失败案例，并能说明验证损失与实际对话质量并不等价。

## 第 4 课：把教学实验扩展到约 100M

**目标**：设计正式实验，而不是把 17M 的教学数据直接放大训练。

1. 先确定目标：简单英文对话、特定领域问答，还是更通用的助手。按目标选择预训练文本和对话微调数据，并单独留出验证集。
2. 为新语料训练或选择合适分词器；约 100M 的基础模型从头预训练后，再用相同词表做 SFT。若改变词表或特殊 token，不能直接加载旧模型权重。
3. 将模型配置从 384/6/6 扩到 README 中约 97.5M 的 768/12/12，先在单卡上测 1、10、100 步的吞吐、显存和损失。现有上下文长度 256 很短，多轮对话需要重新规划上下文长度及其显存成本。
4. 用固定的未见对话题集评估；同时观察预训练与 SFT 的损失、输出格式和事实错误。根据结果再决定是否扩数据或训练步数。

**完成标准**：提交一页实验计划，写明目标、数据来源与许可、token 预算、模型配置、算力预算、验证集和停训条件。

## 文件说明

- `chat_data.py`：读取 JSONL 对话、编码、构造仅覆盖助手回答的标签。
- `chat_sft.py`：`inspect`、`train`、`chat` 三个实验命令。
- `data/chat_demo_train.jsonl`、`data/chat_demo_valid.jsonl`：小型教学数据。

当前实验沿用 TinyStories 的英文词表，因此教学数据也用简短英文。若想训练中文对话模型，需要重新选中文或多语语料并规划分词器；仅把英文样例翻译成中文不能解决底座的语言能力问题。
