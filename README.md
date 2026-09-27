# 从零实现一个约 17M 参数的语言模型

这是一个用 PyTorch 从零搭建的小型自回归语言模型，使用 TinyStories 数据训练。模型包含自写的 BPE 分词器、RoPE、因果注意力、SwiGLU 和 RMSNorm；训练脚本支持验证集评估、余弦学习率调度与保存最佳检查点。当前默认配置有 16,913,280 个参数。

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

当前已完成的目标是端到端实现并训练 17M 模型。把同一架构改到约 100M，主要是增加宽度和层数；这能检验训练脚本的扩展能力，但本身没有引入新的建模机制。本仓库仍以已实训的 17M 配置为默认值。

| 配置 | 宽度 / 层数 / 注意力头 | 参数量 | 状态 |
| --- | --- | ---: | --- |
| 默认模型 | 384 / 6 / 6 | 16,913,280 | 已训练；训练与验证链路已跑通 |
| 扩展参考 | 768 / 12 / 12 | 97,536,768 | 仅核算参数量，未训练或测速 |

参数量按当前代码计算：令 `V` 为词表大小、`d` 为宽度、`L` 为层数、`h = 8d // 3` 为 SwiGLU 隐藏宽度，则总量为 `2Vd + d + L(4d² + 3dh + 2d)`。词嵌入与输出头是两组独立参数。当前词表大小为 8192。

本地 TinyStories 训练缓存约有 4.60 亿 token。这个数据量足以定义一次扩展试验，但仅凭参数量或 token 数不能推断 100M 模型的生成质量；若以后以质量为目标继续做，应同时检查训练数据、验证结果和所需训练算力，而不是只换更大的数据集。

97.5M 模型的 FP32 参数、梯度及 AdamW 两组状态仅按每参数 16 字节估算约需 1.45 GiB；实际训练还需要激活、logits、CUDA 工作区等内存，须用真实 micro-batch 测峰值。单卡放得下时，无需为了参数量本身引入分布式训练；多卡 DDP 的主要用途是提高吞吐，不能减少每卡完整模型副本的内存需求。当前仓库没有 DDP/FSDP 实现，也没有 100M 的训练结果。

## 文本生成

训练完成后，在项目根目录运行：

```powershell
python generate.py
```

输入英文提示词，脚本会读取 `checkpoints/best_model.pt` 及其中的分词器配置并生成续文。采样参数和最大生成 token 数可在 `generate.py` 中调整。

## 测试

```powershell
python -m unittest discover -s model -p "test_*.py"
python -m unittest discover -s tokenizer -p "test_*.py"
```
