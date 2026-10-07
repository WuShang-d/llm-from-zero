# 从零实现一个约 17M 参数的语言模型

这是一个用 PyTorch 从零搭建的小型自回归语言模型，使用 TinyStories 数据训练。模型包含自写的 BPE 分词器、RoPE、因果注意力、SwiGLU 和 RMSNorm；训练脚本支持验证集评估、余弦学习率调度与保存最佳检查点。当前默认配置有 16,913,280 个参数。

## 项目一览

| 阶段 | 做了什么 | 已有证据 |
| --- | --- | --- |
| 从零预训练 | 手写 8192 词 BPE 和 Transformer；在 TinyStories 上训练 16.9M 模型，在 FineWeb-Edu 上训练 97.5M 模型 | 训练与验证记录见 [实验结果](RESULTS.md) |
| 对话 SFT | 在手写模型上实现对话编码、回答标签掩码和微调 | 17M/100M 的损失与失败案例见 [学习记录](CHAT_LEARNING_RECORD.md) |
| 标准工具链练习 | 对 Qwen3-0.6B-Base 做全参数 SFT，以留出集比较 Base/SFT | [Qwen 实验](qwen_sft/README.md)与 [结果摘要](RESULTS.md) |

**结论：**这个项目重点是理解并验证从分词到对话微调的链路。手写模型的问答质量仍不稳定，不能作为实用助手使用；Qwen SFT 的留出集 token 指标改善也不能直接代表事实正确率。[实验结果](RESULTS.md)列出具体数值、生成样例和限制。

**最快体验：**安装下方依赖并运行 `python -m unittest discover -s model -p "test_*.py"` 和 `python -m unittest discover -s tokenizer -p "test_*.py"`。训练数据与大权重不入库；获得数据后可按后文运行训练，已有检查点时可用 `python generate.py --checkpoint <path> --prompt "Once upon a time"` 生成。目录 [qwen_sft/](qwen_sft/README.md)记录了独立的 Qwen 实验流程。

## 环境准备

需要 Python 3.10 或更新版本。训练脚本会依次选择 CUDA、Apple MPS 或 CPU。在 Windows 上使用 NVIDIA 显卡时，请先按 PyTorch 官网针对你的 CUDA 环境的说明安装 PyTorch，再安装其余依赖：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` 包含 PyTorch、NumPy 和 tqdm。若已有适合本机 CUDA 的 PyTorch，安装依赖时请确认它没有被替换为 CPU 版本。

## 准备数据和训练

将 TinyStories 的训练集、验证集文本放到 `data/`，文件名分别为 `TinyStories-train.txt` 和 `TinyStories-valid.txt`。数据文件、编码缓存和模型检查点不会提交到仓库。

### 约 100M 实验：流式抽取 FineWeb-Edu

下载脚本放在 `data/download_fineweb.py`，默认输出保存在 `data/fineweb_edu/`，不会下载完整的约 550 GB 数据集。先运行 `pip install -r requirements.txt` 安装新增的 `datasets` 依赖，再从项目根目录试取一小份：

```bash
python data/download_fineweb.py --train-tokens 5000000 --valid-tokens 50000 --output-dir data/fineweb_edu_smoke
```

上面是小规模试取。确认网络、磁盘空间和样本内容后，正式抽取约 20 亿训练 token：

```bash
python data/download_fineweb.py
```

脚本以流式方式读取 `HuggingFaceTB/smollm-corpus` 的 `fineweb-edu-dedup` 子集，按文档 ID 稳定划分训练/验证，保存为 `train.txt`、`valid.txt` 和 `manifest.json`。预算采用数据集元数据中的 **GPT-2 token 数**；训练新的分词器后，必须重新统计实际 token 数。输出文件被 Git 忽略。已有输出或中断留下的 `.part` 文件不会被自动覆盖。

`train.py` 默认仍读取 TinyStories 和旧词表；只有显式选择 `fineweb100m` 配置才会读取新语料。新语料下载完成后，还需训练新词表并生成 token 缓存；后续 SFT 必须使用同一词表。

### 使用 FineWeb-Edu 训练约 100M 模型

`fineweb100m` 配置使用 `data/fineweb_edu/{train,valid}.txt`、独立的 8192 词 BPE、`768 / 12 / 12` 模型和 512 token 上下文。先只用训练集前 5000 万字符训练词表（下载时已按文档打乱）：

```bash
python -m tokenizer.train_fineweb_bpe
```

新词表保存在 `tokenizer/fineweb_edu_8k/`，不会覆盖 TinyStories 的根目录词表。新旧词表的 token ID 映射不同，不能拿 17M 检查点直接继续训练。随后生成独立的 token 缓存；这一步会读取完整语料，可能耗时较长：

```bash
python train.py --profile fineweb100m --prepare-data-only
```

在有兼容 CUDA 的 PyTorch 环境中，先分别测 1、10、100 次参数更新（短程试跑不保存权重）：

```bash
python train.py --profile fineweb100m --max-steps 1 --max-val-batches 1
python train.py --profile fineweb100m --max-steps 10 --max-val-batches 1
python train.py --profile fineweb100m --max-steps 100 --max-val-batches 1
```

确认显存、吞吐与损失正常后，正式训练一轮：

```bash
python train.py --profile fineweb100m
```

默认 micro-batch 为 4、梯度累积为 8；可用 `--batch-size`、`--grad-accum-steps` 和 `--block-size` 调整。训练中每 1000 次参数更新覆盖保存 `checkpoints/fineweb_100m/latest_model.pt`，整轮验证后保存 `best_model.pt`。当前脚本不支持从 `latest_model.pt` 自动恢复优化器进度；长时间训练前应先完成短程测速并据此估算总时间。

仓库已包含 `merges.txt` 和 `vocab.json`，默认直接使用现有的 8192 词表。首次运行会按需生成 `data/token_cache/` 中的 token 文件；如果数据文件或分词规则发生变化，缓存会重建。若要从训练文本重新训练 BPE，可把 `train.py` 中的 `train_bpe_tokenizer` 改为 `True`。

在项目根目录运行：

```powershell
python train.py
```

默认模型配置：`d_model=384`、6 层 Transformer、6 个注意力头、序列长度 256、批量大小 32、训练 1 个 epoch。训练超参数直接在 `train.py` 顶部修改。每轮结束后评估验证集，较优模型保存为 `checkpoints/best_model.pt`；`checkpoint_every` 控制额外的定期检查点。

### 训练性能选项

`--batch-size` 是每次前向/反向传播处理的样本数，`--grad-accum-steps` 是累积多少次后更新参数；有效 batch 等于两者的乘积。例如在显存较紧时，可用下面的配置保持有效 batch 为 32：

```bash
python train.py --batch-size 4 --grad-accum-steps 8 --amp auto
```

`--amp auto` 在 CUDA 上优先使用 bf16，不支持时使用 fp16；CPU/MPS 上保持 fp32。也可使用 `--amp off` 关闭混合精度。fp16 使用 GradScaler，梯度裁剪在反缩放后、参数更新前执行。默认累积次数为 1。

每轮会输出训练阶段的 **tokens/s**（实际处理的目标 token 数除以训练耗时）和 **peak CUDA memory**（PyTorch 在该轮训练期间的峰值已分配显存）。计时包含数据读取，不包含验证或保存检查点；非 CUDA 环境显示 `N/A`。这两个数用于比较同一设备上不同 batch、精度和模型配置，不代表跨设备的固定性能。

短程检查训练链路可运行：

```bash
python train.py --batch-size 1 --grad-accum-steps 2 --max-steps 2 --max-val-batches 1
```

使用 `--max-steps` 或 `--max-val-batches` 时自动禁用检查点保存，避免短程结果覆盖正式模型。短程验证 loss 只用于检查程序运行，不能与完整验证集结果直接比较。

## 项目总结与约 100M 参数的扩展边界

已经端到端训练 17M TinyStories 模型，也完成了 97.5M FineWeb-Edu 预训练和小规模领域 SFT；具体记录与模型限制见 [实验结果](RESULTS.md)。把同一架构改到约 100M，主要是增加宽度和层数，本身没有引入新的建模机制。本仓库仍以 17M 配置为训练脚本默认值。

| 配置 | 宽度 / 层数 / 注意力头 | 参数量 | 状态 |
| --- | --- | ---: | --- |
| 默认模型 | 384 / 6 / 6 | 16,913,280 | 已训练；训练与验证链路已跑通 |
| FineWeb-Edu 模型 | 768 / 12 / 12 | 97,536,768 | 已完成一轮预训练及领域 SFT；问答效果仍不稳定 |

参数量按当前代码计算：令 `V` 为词表大小、`d` 为宽度、`L` 为层数、`h = 8d // 3` 为 SwiGLU 隐藏宽度，则总量为 `2Vd + d + L(4d² + 3dh + 2d)`。词嵌入与输出头是两组独立参数。当前词表大小为 8192。

本地 TinyStories 训练缓存约有 4.60 亿 token。仅凭参数量或 token 数不能推断模型的生成质量；已有实验的实际输出见 [结果页](RESULTS.md)。

97.5M 模型的 FP32 参数、梯度及 AdamW 两组状态仅按每参数 16 字节估算约需 1.45 GiB；实际训练还需要激活、logits、CUDA 工作区等内存，须用真实 micro-batch 测峰值。单卡放得下时，无需为了参数量本身引入分布式训练；多卡 DDP 的主要用途是提高吞吐，不能减少每卡完整模型副本的内存需求。当前仓库没有 DDP/FSDP 实现。

## 文本生成

FineWeb 100M 基础模型训练完成后，在项目根目录运行：

```powershell
python generate.py --prompt "The history of computing begins"
```

默认读取 `checkpoints/fineweb_100m/best_model.pt`，并从检查点加载对应的 8192 词 BPE 和 512 token 上下文配置。省略 `--prompt` 可进入连续输入的交互模式，输入 `/exit` 退出。可通过 `--max-new-tokens`、`--temperature`、`--top-k` 和 `--seed` 调整生成；例如 `--temperature 0` 使用确定性的贪心解码。该模型只有预训练，适合英文续写，尚不具备稳定的对话指令跟随能力。旧 TinyStories 模型仍可用 `python generate.py --profile tinystories` 加载。

## 测试

```powershell
python -m unittest discover -s model -p "test_*.py"
python -m unittest discover -s tokenizer -p "test_*.py"
```
