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
