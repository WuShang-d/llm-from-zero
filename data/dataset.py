import numpy as np
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    # block_size 就是序列长度 T
    def __init__(self, token_ids, block_size):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index):
        start = index * self.block_size
        end = start + self.block_size + 1

        chunk = self.tokens[start:end]

        # shape: [T]
        x = chunk[:-1]
        y = chunk[1:]

        return x, y


class MemmapLanguageModelDataset(Dataset):
    """从磁盘上的 uint16 token 文件按需读取训练块。"""

    def __init__(self, token_path, block_size):
        self.tokens = np.memmap(token_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

        if len(self.tokens) <= block_size:
            raise ValueError(
                f"token 数量 {len(self.tokens)} 必须大于 block_size {block_size}"
            )

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index):
        start = index * self.block_size
        end = start + self.block_size + 1
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)
        chunk = torch.from_numpy(chunk)
        return chunk[:-1], chunk[1:]
