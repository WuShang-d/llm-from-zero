import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from tokenizer.bpe import (
    BPEEncoder,
    assign_special_ids,
    build_vocab,
    get_merges,
    save_merges,
    save_vocab,
    train_bpe,
)


def create_cosine_scheduler(
    optimizer,
    total_steps,
    min_lr_ratio,
    warmup_ratio=0.05,
    max_warmup_steps=500,
):
    warmup_steps = min(
        max_warmup_steps,
        max(1, int(total_steps * warmup_ratio)),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    progress = tqdm(loader, desc="Validating", unit="batch", leave=False)
    for x, y in progress:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )

        token_count = y.numel()
        total_loss += loss.item() * token_count
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_tokens += token_count

        progress.set_postfix(loss=f"{loss.item():.4f}")

    model.train()
    return total_loss / total_tokens, total_correct / total_tokens * 100


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_loss,
    model_config,
    tokenizer_config,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "model_config": model_config,
            "tokenizer_config": tokenizer_config,
        },
        path,
    )


def read_text_prefix(path: Path, max_chars: int) -> str:
    with path.open("r", encoding="utf-8") as file:
        return file.read(max_chars)


def prepare_tokenizer(
    train_path: Path,
    special_tokens,
    vocab_size,
    train_new,
    sample_chars,
):
    if train_new:
        corpus_sample = read_text_prefix(train_path, sample_chars)
        corpus_sample = corpus_sample.replace("<|endoftext|>", "<|eos|>")
        merges = train_bpe(
            corpus_sample,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )
        save_merges(merges)
    else:
        merges = get_merges()

    special_ids = assign_special_ids(merges, special_tokens)
    vocab = build_vocab(merges, special_ids)

    if train_new:
        save_vocab(vocab, special_ids)

    return merges, special_ids, vocab


def _cache_fingerprint(source_path, merges, special_ids):
    source_stat = source_path.stat()
    payload = {
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "merges": merges,
        "special_ids": special_ids,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_token_cache(
    source_path: Path,
    token_path: Path,
    merges,
    special_ids,
):
    """逐行编码文本，并缓存为 uint16 token；源文件或 tokenizer 变化时重建。"""
    if max(special_ids.values()) >= np.iinfo(np.uint16).max:
        raise ValueError("uint16 token cache 只支持 vocab_size <= 65535")

    metadata_path = token_path.with_suffix(token_path.suffix + ".json")
    fingerprint = _cache_fingerprint(source_path, merges, special_ids)

    if token_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return metadata["token_count"]

    token_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = token_path.with_suffix(token_path.suffix + ".tmp")
    encoder = BPEEncoder(merges, special_ids)
    eos_id = special_ids["<|eos|>"]
    token_count = 0

    with (
        source_path.open("r", encoding="utf-8") as source,
        temporary_path.open("wb") as target,
        tqdm(
            total=source_path.stat().st_size,
            desc=f"Tokenizing {source_path.name}",
            unit="B",
            unit_scale=True,
        ) as progress,
    ):
        for line in source:
            if line.strip() == "<|endoftext|>":
                token_ids = [eos_id]
            else:
                token_ids = encoder.encode(line)

            np.asarray(token_ids, dtype=np.uint16).tofile(target)
            token_count += len(token_ids)
            progress.update(len(line.encode("utf-8")))

    temporary_path.replace(token_path)
    metadata_path.write_text(
        json.dumps(
            {"fingerprint": fingerprint, "token_count": token_count},
            indent=2,
        ),
        encoding="utf-8",
    )
    return token_count
