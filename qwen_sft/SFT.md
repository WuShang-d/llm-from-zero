## 1. 准备数据：`prepare_qwen_sft.py`

输入是一小批来源清楚的“问题—标准回答”。输出三个文件：`train.jsonl`、`valid.jsonl`、`test.jsonl`。每条训练数据先采用这一种格式：

```json
{
  "prompt": [{"role": "user", "content": "用一句话解释什么是梯度下降。"}],
  "completion": [{"role": "assistant", "content": "梯度下降是一种沿损失函数下降方向逐步更新参数的方法。"}]
}
```

代码至少检查：字段和角色顺序正确、问题和回答非空、重复题没有跨集合出现、长度没有超过你设置的训练上限。测试题只用于最后比较回答，不传给训练器。

## 2. 检查一条样本：`inspect_qwen_sft.py`

这是本课**最重要的代码**。加载 `Qwen/Qwen3-0.6B-Base` 自带的 tokenizer，把一条样本交给准备使用的 TRL 数据处理流程，然后打印：

- 聊天模板格式化后的完整文本；
- 每个位置的 token ID 和可读 token；
- 每个位置的 label：`-100` 表示不计算 loss，其他值表示要预测该 token；
- 实际参与 loss 的文本，以及回答结束标记。

检查结果应当能说明：用户问题提供上下文，**回答 token 才是监督目标**。不要仅凭原始 JSON 的字段推断标签；要看处理后真正送进 batch 的 `labels`。TRL 文档说明，`prompt`/`completion` 数据默认只监督 completion；建议在配置里也显式写 `completion_only_loss=True`，并先关闭 `packing`，方便逐 token 核对。[TRL 文档](https://huggingface.co/docs/trl/sft_trainer)

## 3. 短跑训练：`train_qwen_sft.py`

用 `SFTTrainer` 加载 Qwen Base、它的 tokenizer、训练集和验证集。先设置小 batch、短序列、少量 `max_steps`，记录训练及验证 loss。代码还要保存：

- 模型和 tokenizer；
- 实际使用的训练配置；
- 数据文件的版本或校验值。

这一阶段才调用 `trainer.train()`。不要把仓库的 8192 词 BPE 或 `chat_sft.py` 里的模型类接进来；它们属于前几课的手写模型。

## 4. 固定题对比：`compare_qwen_sft.py`

读取**同一份未见过的测试题**，分别让原始 Base 和训练后的模型回答。两边使用相同的聊天模板、提示文本和解码参数，输出并排结果。除了看 loss，也要看回答是否真正更符合任务。

**推荐的编写顺序是：数据文件 → 单条样本标签检查 → Base 回答记录 → 短跑训练 → 同题对比。** 其中第 2 步通过之前，不必开始训练。最终交作业时，最有说服力的材料是“一条样本的格式化文本、token/label 表，以及 Base 与 SFT 的固定题回答对比”。

## 5. 留出集量化测试与交互体验

`test_qwen_sft.py --include-base` 使用训练时相同的 prompt/completion 处理方式，
在 `test.jsonl` 上分别计算 Base 和 SFT 的 completion loss、困惑度和逐 token 命中率。
这些指标衡量对参考回答的预测能力；开放式回答的事实正确性仍需看
`compare_qwen_sft.py` 保存的并排输出。`generate_qwen_sft.py` 提供交互式输入，
方便亲自试用训练后的模型。运行命令见 `README.md`。
