# 从续写模型到对话模型：分阶段实验教案

这套实验先复用现有的约 17M 参数 TinyStories 检查点和 8192 词表，把对话训练链路跑通。自带的 24 条训练对话、4 条验证对话是**教学样例**，只用于检查格式、损失和生成流程；反复训练会记住样例，不能据此判断通用对话能力。不会改写原来的 `train.py`、`generate.py` 或 `checkpoints/best_model.pt`。

## 先明确两条学习线

| 阶段 | 用什么模型 | 重点学什么 | 交付物 |
| --- | --- | --- | --- |
| 1–3 课 | 已训练的 17M | 对话编码、标签错位、损失掩码、SFT、生成与评估 | 一份包含损失曲线和失败案例的实验记录 |
| 第 4 课 | **完成预训练之后**的约 97.5M | 从头训练的预算、稳定性、检查点、数据与模型规模的影响 | 与 17M 使用同一批原始问答的对比报告 |
| 第 5–6 课 | 开源预训练 Base 模型，例如 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 标准工具链、聊天模板、全量 SFT 与 LoRA | Base、SFT、LoRA 的同题对比 |
| 第 7 课（进阶） | 已做 SFT 的开源模型 | 成对偏好数据与 DPO | 偏好数据审查及小规模对照实验 |

17M 和 100M 的主线是**弄懂机制**；0.6B Base 的主线是**学习实用工作流**。两种模型的架构、词表和预训练数据不同，跨模型比较回答时可以用相同的原始问题，但不要直接比较它们的 token 级 loss 或困惑度。以下命令中，前四课使用本仓库现成脚本；第 5–7 课是学习和实现路线，仓库目前没有对应的 Qwen/TRL 训练脚本。

## 第 1 课：数据怎样变成模型看到的 token

**目标**：理解 `User:`、`Assistant:`、`<|bos|>`、`<|eos|>` 的作用，以及下一 token 预测中的输入与标签错位。

```bash
cd ~/Documents/llm-from-zero
python chat_sft.py inspect
```

阅读 `data/chat_demo_train.jsonl` 和 `chat_data.py`。自己新增一条两轮对话，再运行 `inspect`。思考：为什么第二轮要保留第一轮作为上下文？

重点读 `encode_conversations()`：它把每个助手轮次做成一个样本，前面的对话是条件。手写一个极短的例子，逐位对齐 `input_ids = full[:-1]` 与 `labels = full[1:]`；再标出 `-100`、回答正文和 `<|eos|>` 的位置。检查 `Assistant: ` 是提示前缀，不是需要预测的回答正文。本项目使用普通文本 `User:`/`Assistant:` 标记角色，并未为角色新增专用 token。

**完成标准**：能指出哪些 token 作为条件输入，哪些 token 被计入损失；能解释为何训练和推理必须使用一致的对话格式。

## 第 2 课：只训练助手回答

**目标**：理解监督微调（SFT）、`ignore_index=-100`、训练集和验证集的区别。

先做不保存权重的链路检查，再运行教学训练：

```bash
python chat_sft.py train --steps 2 --eval-every 1 --no-save
python chat_sft.py train --steps 40 --eval-every 10
```

结果单独存为 `checkpoints/chat_demo.pt`。记录训练损失与验证损失。试着把 `--steps` 增大，观察训练损失继续下降时验证损失是否也下降。

重点学交叉熵、teacher forcing、batch、学习率、梯度裁剪和过拟合。读 `supervised_loss()`、`validation_loss()`：`-100` 不参与交叉熵；验证 loss 按有效回答 token 数加权。数据文件有 24 条训练对话、4 条验证对话；按助手轮次展开后是 28 个训练样本、5 个验证样本，数值波动很大。做一个小消融：固定其余条件，只改变训练步数，记录 `step / train_loss / valid_loss / 检查点`。不要把一次短程验证或训练集上的低 loss 当作泛化证据。

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

建立固定题集，至少包含问候、简单事实、改写、多轮追问和超出 TinyStories 能力范围的问题。比较时固定原始提示、`--temperature 0` 和 `--max-new-tokens`，记录原文输出；人工标注“是否回答问题 / 是否按格式结束 / 是否编造 / 是否重复”。可以算每类通过数，但五题样本太小，不应把它写成可靠的通用能力分数。再用一个提示比较不同温度，理解采样变化与训练效果是两回事。

**完成标准**：写出至少两个改进和两个失败案例，并能说明验证损失与实际对话质量并不等价。

## 第 4 课：把教学实验扩展到约 100M

**目标**：设计正式实验，而不是把 17M 的教学数据直接放大训练。

1. 先确定目标：简单英文对话、特定领域问答，还是更通用的助手。按目标选择预训练文本和对话微调数据，并单独留出验证集。
2. 为新语料训练或选择合适分词器；约 100M 的基础模型从头预训练后，再用相同词表做 SFT。若改变词表或特殊 token，不能直接加载旧模型权重。
3. 将模型配置从 384/6/6 扩到 README 中约 97.5M 的 768/12/12；`fineweb100m` 配置使用 512 token 上下文。先按 README 测 1、10、100 步的吞吐、显存和损失。多轮对话仍可能超过上下文长度，需要检查截断与显存成本。
4. 用固定的未见对话题集评估；同时观察预训练与 SFT 的损失、输出格式和事实错误。根据结果再决定是否扩数据或训练步数。

这一课应补上预训练基础：BPE 对不同语言的切分、训练/验证数据隔离、token 预算、有效 batch（micro-batch × 梯度累积）、学习率调度、混合精度、梯度爆炸与 NaN、吞吐和显存。每次试跑记录配置、数据版本、随机种子、有效 token 数、loss、tokens/s、峰值显存及实际耗时。检查点要区分“能加载权重”和“能恢复优化器及训练进度”：目前 `train.py` **不能从 `latest_model.pt` 自动续训**。

100M 预训练确实完成并选好 Base 检查点后，才能用 `chat_sft.py train --base checkpoints/fineweb_100m/best_model.pt --output checkpoints/fineweb_100m/chat_demo.pt` 做同类教学 SFT。训练前先用该检查点运行 `chat_sft.py inspect --base ...` 检查词表和样本长度；100M 必须沿用它自己的分词器，不能拿 17M 的 token ID、嵌入权重或旧 token 缓存混用。用同一批**原始文本**问题比较 17M/100M 的 Base 与 SFT 四种输出，注明它们的预训练语料也不同，因此结果不能只归因于参数量。

**完成标准**：提交一页实验计划，写明目标、数据来源与许可、token 预算、模型配置、算力预算、验证集和停训条件。

## 第 5 课：用开源 Base 模型做标准 SFT

**目标**：学会把本仓库手写的编码与损失逻辑，对应到 Transformers/TRL 的数据格式和训练配置。开源 Base 模型已经经过预训练，但 Base 并不等于指令模型；先记录未经 SFT 的回答作为对照。

1. 阅读 [Qwen3-0.6B-Base 模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-Base)，确认模型版本、许可、模型所需的 Transformers 版本、原生词表与聊天模板。使用该模型自带分词器；不要移植本仓库 8192 词 BPE。
2. 先用一小批有合法来源、语言和任务目标一致的指令—回答样本。分开训练、验证和固定测试题；检查重复样本、答案质量、过长截断、角色顺序和模板结束标记。可以先把一条数据转成 `{"prompt": [...], "completion": [...]}`，逐 token 检查被监督的部分。
3. 阅读 [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer)。弄清 `messages` 数据与 `prompt`/`completion` 数据的差别：前者的默认 loss 范围可能包含整段，后者默认只覆盖 completion；如使用 `assistant_only_loss=True`，还要核对聊天模板能产生助手掩码。不要仅凭字段名猜测 loss 掩码。
4. 从小样本、短步数开始，保存模型、分词器、训练配置与数据版本；逐步检查 loss、未见问题上的回答和显存。固定提示和解码参数，将 Base 与 SFT 模型并排评估。

**完成标准**：能展示一条样本的格式化文本、token、有效 label 区间，并解释为什么同一份 JSONL 直接喂给不同模型可能得到不同训练目标。

## 第 6 课：在同一个 Base 上学习 LoRA

**目标**：理解冻结底座、只训练低秩适配器的含义及其资源和效果取舍。阅读 [PEFT LoRA 文档](https://huggingface.co/docs/peft/main/conceptual_guides/lora)与 [TRL 的 PEFT 集成](https://huggingface.co/docs/trl/sft_trainer#train-adapters-with-peft)。

先检查目标模块名称，再选择 `target_modules`；记录秩 `r`、`lora_alpha`、可训练参数数目、显存、耗时及适配器文件大小。用相同 Base、数据划分和固定测试题，将 LoRA 与第 5 课的全量 SFT 对比。加载适配器推理时仍需正确的底座模型与分词器；确认加载后结果与保存前一致。资源不够时可以只做 LoRA，但要如实标注没有全量 SFT 对照。

**完成标准**：能说明 LoRA 更新了哪些权重、为什么适配器较小，以及“可训练参数更少”不等于训练显存按同样比例下降。

## 第 7 课（进阶）：理解偏好数据和 DPO

**目标**：在已有 SFT 模型后，理解 DPO 与 SFT 的不同监督信号。先读 [TRL DPOTrainer 文档](https://huggingface.co/docs/trl/dpo_trainer)。每条偏好数据包含**相同 prompt** 下的 `chosen` 和 `rejected` 两个回答；偏好依据要明确，不能只根据长度或个人印象随意标注。

先人工审查少量成对样本，检查两个回答是否都针对同一问题、是否存在明显标签错误或测试集泄漏。再在相同 SFT 起点上做小规模 DPO，记录参考模型设置、偏好 loss、未见题上的胜率与失败案例。DPO 的训练目标涉及相对于参考模型的偏好差异；偏好 loss 下降不自动证明真实帮助性提高。自研 17M/100M 若要接入 TRL/PEFT，还需适配模型、分词器、配置、保存和对话模板接口，这可作为后续独立练习。

**完成标准**：能写出一条合格偏好样本，说明为何应先有 SFT 起点，并用固定题集展示 DPO 前后具体变化。

## 每次实验都保留什么记录

至少保存：实验目标、代码版本、数据来源及划分、模型与分词器、对话模板、随机种子、训练步数与有效 token 数、优化器和学习率、batch 与上下文长度、验证损失、显存和耗时、固定测试题的原始输出。遇到 NaN、截断、重复生成或忘记结束标记，先检查数据、掩码和生成配置，再调整训练时长。不要把训练集样例、验证集或公开基准题混进最终测试题集。

## 文件说明

- `chat_data.py`：读取 JSONL 对话、编码、构造仅覆盖助手回答的标签。
- `chat_sft.py`：`inspect`、`train`、`chat` 三个实验命令。
- `data/chat_demo_train.jsonl`、`data/chat_demo_valid.jsonl`：小型教学数据。

当前实验沿用 TinyStories 的英文词表，因此教学数据也用简短英文。若想训练中文对话模型，需要重新选中文或多语语料并规划分词器；仅把英文样例翻译成中文不能解决底座的语言能力问题。
