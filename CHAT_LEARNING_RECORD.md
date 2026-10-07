# 从续写模型到对话模型：分阶段实验教案与学习记录

这套实验先复用现有的约 17M 参数 TinyStories 检查点和 8192 词表，把对话训练链路跑通。自带的 24 条训练对话、4 条验证对话是**教学样例**，只用于检查格式、损失和生成流程；反复训练会记住样例，不能据此判断通用对话能力。原有的 `train.py`、`generate.py` 和 `checkpoints/best_model.pt` 不作为这次对话微调的输出。

## 先明确两条学习线

| 阶段 | 用什么模型 | 重点学什么 | 交付物 |
| --- | --- | --- | --- |
| 1–3 课 | 已训练的 17M | 对话编码、标签错位、损失掩码、SFT、生成与评估 | 一份包含损失记录和失败案例的实验记录 |
| 第 4 课 | **完成预训练之后**的约 97.5M | 从头训练的预算、稳定性、检查点、数据与模型规模的影响 | 与 17M 使用同一批原始问答的对比报告 |
| 第 5–6 课 | 开源预训练 Base 模型，例如 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 标准工具链、聊天模板、全量 SFT 与 LoRA | Base、SFT、LoRA 的同题对比 |
| 第 7 课（进阶） | 已做 SFT 的开源模型 | 成对偏好数据与 DPO | 偏好数据审查及小规模对照实验 |

17M 和 100M 的主线是理解机制；0.6B Base 的主线是学习实用工作流。跨模型可比较相同原始问题的回答，但不直接比较不同词表下的 token 级 loss 或困惑度。

## 第 1–3 课：已完成的实践

### 第 1 课：对话编码与标签

- 一条 JSONL 记录可以包含多轮 `user`/`assistant` 对话。`encode_conversations()` 按每个助手轮次生成一个训练样本，较早的轮次作为上下文。
- `full[:-1]` 是输入，`full[1:]` 是右移一位的目标；因此长度检查使用 `len(full) - 1`，减去的不是 `<|eos|>` 的长度。
- `User:` 和 `Assistant: ` 是普通文本形式的角色提示。当前助手回答及结束标记 `<|eos|>` 是需要预测的部分；上下文与提示前缀在 `labels` 中标为 `-100`，由 `ignore_index=-100` 排除在损失之外。批量补齐的标签也使用 `-100`。

### 第 2 课：100 步 SFT 与损失记录

运行命令：

```bash
uv run python chat_sft.py train --steps 100 --eval-every 10
```

运行环境和样本数：`device=mps`、训练样本 28、验证样本 5、最大输入长度 256 token。下面是这次运行的原始数值；`train_loss` 是评估时所在训练 batch 的损失，`valid_loss` 是验证集上按有效回答 token 加权的损失。

| Step | Train loss | Valid loss | 保存最佳检查点 |
| ---: | ---: | ---: | :---: |
| 10 | 2.1135 | 2.5815 | 是 |
| 20 | 1.2197 | 2.1854 | 是 |
| 30 | 1.1926 | 2.1099 | 是 |
| 40 | 0.3762 | 2.0982 | 是 |
| **50** | **0.2534** | **2.0869** | **是** |
| 60 | 0.0514 | 2.1444 | 否 |
| 70 | 0.0277 | 2.2173 | 否 |
| 80 | 0.3108 | 2.3018 | 否 |
| 90 | 0.0098 | 2.2796 | 否 |
| 100 | 0.0073 | 2.3194 | 否 |

**观察：**验证损失在第 50 步达到本次记录的最低值 2.0869，之后总体回升，到第 100 步为 2.3194；与此同时，训练 batch 损失总体降至接近 0。这是本次小样本实验中明显的过拟合迹象。第 80 步的训练损失反弹表明单个 batch 的数值会波动，不能把这一列当作平滑的全训练集曲线。`checkpoints/chat_demo.pt` 只在验证损失刷新最低值时覆盖保存，因此本次运行最后保存的是第 50 步的检查点，并非第 100 步权重。验证样本只有 5 个，最低点和数值差异不宜被解释成稳定的泛化能力估计。

### 第 3 课：多轮生成观察

下面是一次交互式运行的原始输出；此处没有给出该次运行对应的检查点版本、温度等完整配置，因此将它作为定性失败案例，不作为可复现的能力评分。

```text
uv run python chat_sft.py chat
User: Name an animal that can fly.
Assistant: A bird can fly in a different way.
User: what is ai
Assistant: Aiquorms can fly.
User: How are you
Assistant: I am a great teacher.
User: What is a cat
Assistant: A cat is a big cat with black and white spots.
User: Tell me about a tree.
Assistant: Could I take a picture of a cat?
```

**观察：**模型能够按 `Assistant:` 格式产出文本，但回答质量不稳定。问 AI 时延续了 `fly`，问 tree 时又回到了 `cat`；对 cat 的解释也不准确。脚本会把每轮用户问题和模型回答都加入下一轮上下文，所以早先话题和模型自己的错误回答可能影响后续生成。不过，仅凭这一段对话无法把错误完全归因于上下文：小模型在没有历史时也可能答非所问。

还要注意，`Name an animal that can fly.` **原样出现在训练数据中**，因此这条回答不能用于判断未见问题上的泛化。当前记录完成了编码、训练和多轮生成链路的实践；若要达到教案中“固定未见题集、Base 与 SFT 对照”的正式评估要求，仍需另做对照实验，并保存检查点、解码参数和原始输出。

## 后续路线

- **第 4 课：约 100M 的领域英文问答。**使用已核对的 FineWeb 100M Base 及匹配词表，准备 AI/LLM/Agent 问答数据、做领域 SFT 和对照评估。具体计划见下文。
- **已完成：Qwen3-0.6B-Base 的全参数 SFT。**使用模型自带分词器与聊天模板，检查有效 label 范围，并在固定留出集比较 Base 与 SFT。结果摘要见 [RESULTS.md](RESULTS.md)，运行步骤见 [qwen_sft/README.md](qwen_sft/README.md)。
- **第 6 课：LoRA。**在同一个 Base、同一数据划分和题集上记录可训练参数、显存、耗时和回答差异。
- **第 7 课：DPO。**在已有 SFT 模型上审查同一 prompt 的 `chosen`/`rejected` 回答，并用未见题比较训练前后表现。

每次实验继续保存命令、代码版本、检查点、分词器、数据划分、随机种子、损失、推理参数和原始输出。

## 第 4 课实验计划：100M 英文 AI/LLM/Agent 问答

### 已确定的目标与模型边界

- **目标：**训练一个能用简短英文回答 AI、LLM、Agent 基础概念问题的领域模型；先检验格式、术语和简单事实，不预设它能成为通用助手。
- **已核对的 Base：**本地 `checkpoints/fineweb_100m/best_model.pt` 与训练机器上的对应检查点一致，包含 97,536,768 个参数，配置为宽度 768、12 层、12 头、8192 词表、512 token 上下文；检查点记录 `epoch=1`、`best_val_loss≈2.7676`。这些数值说明检查点可用于 Base 对照，不代表它已经具备 AI/LLM/Agent 领域问答能力。
- **预训练语料已核对：**启动参数为 `--profile fineweb100m`。该 profile 选择 `data/fineweb_edu/train.txt`、`data/fineweb_edu/valid.txt`、`data/fineweb_edu_token_cache/` 和 `tokenizer/fineweb_edu_8k/`；`TinyStories-train.txt` 只属于默认 `tinystories` 分支。因此这次 100M 预训练使用的是 **FineWeb-Edu 文本，不是本地 `data/TinyStories-train.txt`**。数据清单记录上游为 `HuggingFaceTB/smollm-corpus` 的 `fineweb-edu-dedup` 配置，按文档 ID 划分训练/验证。检查点本身不携带语料路径；这里的结论由启动参数、代码路径、数据清单和对应缓存共同支持。
- **已提供的预训练运行记录：**CUDA、micro-batch 4、梯度累积 8、有效 batch 32、bfloat16；训练 2,373,043,187 token，验证 23,772,234 token；训练 batch 1,158,712，验证 batch 11,608。用户贴出的进度是第 1/1 epoch 的第 96 个 batch，loss 9.4036、学习率约 `7.80e-06`，属于训练初期快照，不能当作最终训练损失。最终选择的检查点记录 `best_val_loss≈2.7676`。
- **上下文与数据限制：**现有 `chat_sft.py` 要求每条训练样本的输入长度不超过 512 token；`encode_conversations()` 对过长样本会报错。先从简短单轮问答开始，再加入少量可容纳的多轮澄清问题。

### 词表和检查点交接

1. 已从训练机器的 `tokenizer/fineweb_edu_8k/merges.txt` 复制到本地 `tokenizer/fineweb_100m_merges.txt`，与 `fineweb_100m_best_model.pt` 使用相同的模型前缀。该文件 SHA-256 为 `73f6f53348d689e192884ff4daf11e681e9e6f658d78d68caa9f11d616bc3ad6`，与训练用文件相同。
2. 用本项目 `get_merges()` 读取后，已确认 7,933 条 merge 的**内容和顺序**与 100M 检查点内嵌的 `tokenizer_config.merges` 完全相同。仓库根目录的 `merges.txt` 虽然也是 7,933 条，却与该检查点不一致，不能混用。
3. `chat_sft.py` 会直接使用检查点内嵌的 `merges` 与 `special_ids`；独立 `fineweb_100m_merges.txt` 用于保存和审计。已用现有教学数据运行 `inspect --base checkpoints/fineweb_100m/best_model.pt`，成功得到 28 个样本；首个样本输入 26 token、有效回答标签 12 token。选定领域数据后仍须对新数据重新检查长度。

### SFT 数据来源与筛选

优先选 **AI/LLM/Agent 基础知识的短英文问答**。已找到直接覆盖 Agent 和机器学习的候选，但它们仍需核对答案事实、长度和文件格式；没有一份可以不经审查直接投入训练。

| 候选 | 已确认的信息 | 对本实验的用途和限制 |
| --- | --- | --- |
| [AYI-NEDJIMI/ai-agents-en](https://huggingface.co/datasets/AYI-NEDJIMI/ai-agents-en) | 132 条英文 Agent 主题记录，字段含 `question`、`answer`、`source_url`；标注 MIT 许可 | **最贴题的起点。**涵盖 Agent 架构、工具调用、多 Agent 和 RAG；规模很小，一些回答涉及具体框架与版本，必须逐条核实，并检查 512 token 窗口。 |
| [Koushim/qa-ml-dl-jsonl](https://huggingface.co/datasets/Koushim/qa-ml-dl-jsonl) | 英文 ML/DL/RL 问答，标注 MIT 许可；数据卡列出两种答案长度 | 可补基础 AI/ML 概念。当前 Hugging Face 数据集预览报错：`answer` 字段有字符串与数组混用，不能直接用默认加载；需按原始 JSONL 逐条解析、规范化并审核答案。 |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | Apache-2.0；含英文等多语的人工对话，消息带角色、父消息 ID、审核与排序字段 | 补少量自然问答和多轮格式。筛英文 AI/LLM/Agent 主题，从对话树重建有效的 `user → assistant` 路径；主题命中数量尚未统计。 |
| [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | 英文 instruction/response 数据，包含 open/closed QA；许可为 CC BY-SA 3.0 | 备选。不是专门的 AI/Agent 数据；closed QA 若依赖给定 `context`，必须把必要上下文一起放入输入。 |
| [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | 约 1.37 万道科学考试问题，许可为 CC BY-NC 3.0 | **领域替代方案**，只有在 AI/LLM/Agent 问答不足时采用。它的选择题和支持文本需要按任务重新组织，不可直接当作自然对话；非商业限制需保留。 |

两份下载文件已复制到 `data/ai_qa/raw/`，原文件仍保留在 `~/Downloads`。其中 `dataset_en.jsonl` 有 132 条 Agent 问答，`brief_questions_with_answers.jsonl` 有 24,215 条 ML/DL 问答。后者的 24,215 条均可解析为 JSON，但主题很宽，且存在重复问题。可复现的结构清洗脚本是 `data/prepare_ai_qa_sft.py`：仅从大数据集中选 LLM 相关主题或明确提到 LLM/语言模型等的问题，去掉空值、特殊 token 标记、过短内容、超过模型窗口的样本，以及同一规范化问题对应不同答案的冲突组；按来源分别以固定种子划分。原始文件哈希、清洗计数和来源许可保存在 `data/ai_qa/manifest.json`，每条保留样本的原始 ID、主题和可用来源链接保存在 `data/ai_qa/provenance.jsonl`。

本次运行结果：大数据集有 23,573 条不属于所选范围、6 条内容过短；合计有 108 条记录落入“同一问题、不同答案”的冲突组，被排除。最终得到 **528 训练 / 66 验证 / 66 测试**，其中 Agent 来源分别为 106 / 13 / 13。三份文件全部通过 `chat_data.encode_conversations()` 检查，最长输入分别为 281 / 242 / 228 token，均低于 512。规范化后的相同问题没有跨集合出现；**语义相近的改写仍可能跨集合**，后续要人工检查。筛选脚本没有验证技术事实，也没有核对 Agent 数据里可能随版本变化的框架描述；正式训练前须抽查并修订答案，同时记录下载文件对应的数据集版本。

转换到本项目时，每行使用 `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`；多轮记录要严格交替。当前代码只对助手回答与 `<|eos|>` 计算损失。转换后逐条运行长度检查；如果添加背景材料，先确认背景与答案总长度仍在目标模型窗口内。先人工建立一组未参与 SFT 的固定英文题，覆盖基础概念、术语区别、简短多轮追问、不确定时承认不知道，以及明显超出所选领域的问题。

### 训练、验证和停训

1. **Base 对照：**用同一组原始问题记录 `checkpoints/fineweb_100m/best_model.pt` 的输出；固定检查点、提示格式、`temperature=0`、`max_new_tokens`，保存原文。之后对 SFT 检查点使用相同设置。
2. **小步试跑：**先用少量已审查的样本运行 `chat_sft.py inspect --base checkpoints/fineweb_100m/best_model.pt --data <SFT数据>`，再试 1–2 步不保存的 SFT，检查 loss 是否有限、输入与有效 label 是否符合预期、MPS/显存与实际耗时是否可承受。现有脚本按检查点配置创建模型，不需要重新运行预训练阶段的 1/10/100 步吞吐测试。
3. **正式 SFT：**在确定数据规模与设备预算后再设步数、batch、学习率和评估间隔；记录随机种子、训练/验证样本数、有效回答 token 数、loss、耗时及峰值内存。按验证损失保存最优检查点，观察是否像 17M 教学实验那样出现过拟合。验证损失只能辅助选点，最终仍以固定未见题的原始输出判断。
4. **停训条件：**验证损失连续多个评估点恶化且未见题回答没有改善，或出现 NaN、持续截断、明显重复与错误事实时，停止并检查数据与配置。若领域题始终不能正确回答，先审查数据质量和 Base 的领域知识；不只靠加大步数补救。

清洗后的文件已保存为 `data/ai_qa/train.jsonl`、`valid.jsonl`、`test.jsonl`；以下命令中的前两步可在人工抽查后运行，正式步数也须在试跑和数据审查后决定。测试集只用于最后评估，不参与训练或选检查点。

```bash
uv run python chat_sft.py inspect --base checkpoints/fineweb_100m/best_model.pt --data data/ai_qa/train.jsonl
uv run python chat_sft.py train --base checkpoints/fineweb_100m/best_model.pt --train-data data/ai_qa/train.jsonl --valid-data data/ai_qa/valid.jsonl --output checkpoints/fineweb_100m/ai_qa_sft.pt --batch-size 1 --steps 2 --eval-every 1 --no-save
# 确认短程试跑正常后，再去掉 --no-save 并按实测决定正式步数与评估间隔。
```

### 首次 100M 领域 SFT 运行记录

实际命令：

```bash
uv run python chat_sft.py train \
  --base checkpoints/fineweb_100m/best_model.pt \
  --train-data data/ai_qa/train.jsonl \
  --valid-data data/ai_qa/valid.jsonl \
  --output checkpoints/fineweb_100m/ai_qa_sft.pt \
  --batch-size 2 --steps 300 --eval-every 25 --lr 2e-5
```

设备 `mps`；训练样本 528，验证样本 66，最大输入长度 512。batch 2 时约 264 步遍历一次训练集；300 步约相当于 1.14 轮。下表保留用户提供的逐点评估数值，训练损失是当步 batch 的值。

| Step | Train loss | Valid loss | 保存新最佳 |
| ---: | ---: | ---: | :---: |
| 25 | 2.3620 | 2.6200 | 是 |
| 50 | 2.2302 | 2.5068 | 是 |
| 75 | 2.5994 | 2.4400 | 是 |
| 100 | 2.1545 | 2.3900 | 是 |
| 125 | 2.3472 | 2.3539 | 是 |
| 150 | 2.2429 | 2.3213 | 是 |
| 175 | 1.9175 | 2.2935 | 是 |
| 200 | 1.8568 | 2.2728 | 是 |
| 225 | 2.7013 | 2.2593 | 是 |
| 250 | 2.3120 | 2.2369 | 是 |
| **275** | **2.4416** | **2.2217** | **是** |
| 300 | 1.5742 | 2.2261 | 否 |

验证损失持续改善到第 275 步，第 300 步仅回升 0.0044；**这一小幅变化不足以断言已经过拟合，也不说明继续训练一定有益**。已直接读取 `checkpoints/fineweb_100m/ai_qa_sft.pt`，其中 `sft_step=275`、`valid_loss=2.2217322051195807`，确认保存的是第 275 步的最佳检查点。训练 batch loss 波动较大，不能拿单个值判断整体趋势。训练日志本身不提供回答准确性的证据，需要看实际生成结果。

### 两次交互式测试：重复与概念混淆

**第一次：**用户用该 SFT 检查点交互，报告了以下原始输出。此前建议的测试命令使用 `temperature=0`，但本次对话未同时保存完整启动命令，因此不将解码参数视为已核实的实验元数据。

```text
User: Hi!
Assistant: Hi! I'm a beginner, but I'm not a beginner. I'm a beginner, but I'm not a beginner. I'm a beginner, but I'm not a beginner. I'm a beginner, but I'm not a beginner. I'm not a begin
User: How can you help me
Assistant: I'm a beginner, but I'm not a beginner. I'm a beginner, but I'm not a beginner. I'm a beginner, but I'm not a beginner. I'm a beginner, but I
User: What is AI
Assistant: AI is a subset of AI that uses machine learning to understand and generate human-like text. It's a subset of AI that uses machine learning to understand and generate human-like text.
User: What is machine learning
Assistant: Machine learning is a subset of AI that uses machine learning to understand and generate human-like text. It's a subset of AI that uses machine learning to understand and generate human-like text.
User: What is ChatGPT
Assistant: ChatGPT is a general-purpose language model that uses machine learning techniques to generate text. It is a powerful tool for language modeling and machine learning.
User: What is Claude
Assistant: ChatGPT is a general-purpose language model that uses machine learning techniques to generate text. It is a powerful tool for language modeling and machine learning.
```

**第二次：**在讨论重复问题后，用户再次交互并报告以下原始输出；完整启动参数同样未随输出保存。

```text
User: Hi
Assistant: Layer 2: Layer 3: Layer 4: Layer 5: Layer 6: Layer 7: Lay
User: What is AI
Assistant: AI is the branch of AI that uses machine learning and natural language processing (NLP) to understand and generate text. The word 'AI'
User: How are you
Assistant: AI is a branch of AI that uses machine learning and natural language processing (NLP) to understand and generate text. It uses machine learning and natural
User: What is ChatGPT
Assistant: AI is a deep bot, like Google, Amazon, or Microsoft, that uses AI models to understand human language, such as text. When it comes
User: What is google
Assistant: Google searches for a specific word in the text in a series of text fields. This allows the model to learn the words that are most relevant to
```

两次运行都能产生 `Assistant:` 形式的文本，但存在重复、答非所问、概念混淆和输出被最大 token 数截断的问题。第一次把 Claude 答成 ChatGPT；第二次连问候都输出了无关的层号。**验证损失下降没有带来可靠的领域问答。**

为排除多轮历史影响，又用该 SFT 检查点在独立进程中分别测试 `Hi!`、`What is AI?`、`What is machine learning?` 和 `What is Claude?`，设置 `temperature=0`、`max_new_tokens=48`。`Hi!` 仍重复 beginner 句，机器学习和 Claude 仍有概念混淆或重复。因此多轮上下文可能放大错误，但**不是这些错误的必要条件**。另一次 `temperature=0.7`、32 token 的机器学习回答虽少了机械重复，事实仍不准确。原始 Base 在同题的贪心生成也发生重复；当前结果不能仅归因于 SFT。

本次 100M 模型作为教学实验到此收尾；不再通过盲目增加步数追求可用的领域助手。若以后需要严格量化，可继续用独立测试集做 Base/SFT 并排评估。接下来转向 Qwen Base 的正式数据准备、训练和评估；缩短输出或调温度都不会自动修复当前小模型的知识与数据质量。

### 与原教案第 3、4 点的关系

原教案里的 1/10/100 步**预训练**吞吐、显存和稳定性测试，是从头训练 100M Base 前的预算检查。目标 Base 已经训练完成，本次 SFT 不必重做这些预训练试跑；SFT 仍需自己的短程链路检查和内存、耗时记录。最后的评估则取决于实际选出的 SFT 数据：固定未见题，比较 Base 与 SFT 的答题相关性、关键事实、对话格式、停止行为及编造情况。17M 教学模型可作为参照，但它与 100M 的预训练数据、词表、架构规模若不同，结果不能只归因于参数量。
