"""Train the language model on the official TinyStories train/validation files."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from data.dataset import MemmapLanguageModelDataset
from data.dataloader import create_dataloader
from model.transformer import TransformerLM
from training_utils import (
    create_cosine_scheduler,
    ensure_token_cache,
    evaluate,
    prepare_tokenizer,
    save_checkpoint,
)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
seed = 42
# 第一次用 8192 词表训练时设为 True；之后可设为 False 直接复用 merges.txt。
train_bpe_tokenizer = False
tokenizer_vocab_size = 8192
# BPE 只需从训练集抽样学习 merge，无需把 1.9 GB 文本全部载入内存。
tokenizer_sample_chars = 50_000_000
block_size = 256
batch_size = 32
num_epochs = 1
checkpoint_every = 10

d_model = 384
num_heads = 6
num_layers = 6

max_learning_rate = 3e-4
min_lr_ratio = 0.1
weight_decay = 0.01
max_grad_norm = 5.0

project_dir = Path(__file__).resolve().parent
train_text_path = project_dir / "data" / "TinyStories-train.txt"
val_text_path = project_dir / "data" / "TinyStories-valid.txt"
token_cache_dir = project_dir / "data" / "token_cache"
checkpoint_dir = project_dir / "checkpoints"

torch.manual_seed(seed)


def main():
    print(f"Train with: {device}")
    special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>"]

    merges, special_ids, vocab = prepare_tokenizer(
        train_path=train_text_path,
        special_tokens=special_tokens,
        vocab_size=tokenizer_vocab_size,
        train_new=train_bpe_tokenizer,
        sample_chars=tokenizer_sample_chars,
    )
    vocab_size = len(vocab)

    train_token_count = ensure_token_cache(
        train_text_path,
        token_cache_dir / "train.bin",
        merges,
        special_ids,
    )
    val_token_count = ensure_token_cache(
        val_text_path,
        token_cache_dir / "valid.bin",
        merges,
        special_ids,
    )

    train_dataset = MemmapLanguageModelDataset(
        token_cache_dir / "train.bin",
        block_size,
    )
    val_dataset = MemmapLanguageModelDataset(
        token_cache_dir / "valid.bin",
        block_size,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size,
        shuffle=False,
        drop_last=False,
    )

    model_config = {
        "vocab_size": vocab_size,
        "num_heads": num_heads,
        "d_model": d_model,
        "num_layers": num_layers,
        "block_size": block_size,
    }
    tokenizer_config = {
        "merges": merges,
        "special_ids": special_ids,
    }
    model = TransformerLM(
        vocab_size=vocab_size,
        num_heads=num_heads,
        d_model=d_model,
        num_layers=num_layers,
    ).to(device)

    optim = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    total_steps = num_epochs * len(train_loader)
    scheduler = create_cosine_scheduler(
        optim,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        f"Train tokens: {train_token_count:,} | "
        f"Validation tokens: {val_token_count:,}"
    )
    print(
        f"Train batches: {len(train_loader):,} | "
        f"Validation batches: {len(val_loader):,} | "
        f"Parameters: {parameter_count:,}"
    )

    for epoch in range(num_epochs):
        train_loss_sum = 0.0
        train_correct = 0
        train_tokens = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )
        for x, y in progress:
            x, y = x.to(device), y.to(device)

            optim.zero_grad(set_to_none=True)

            logits = model(x)  # [B, T, V]
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                y.reshape(B * T),
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            optim.step()
            scheduler.step()

            with torch.no_grad():
                y_pred = logits.argmax(dim=-1)
                train_correct += (y_pred == y).sum().item()

            token_count = y.numel()
            train_loss_sum += loss.item() * token_count
            train_tokens += token_count

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}",
                grad=f"{grad_norm.item():.2f}",
            )

        train_loss = train_loss_sum / train_tokens
        train_acc = train_correct / train_tokens * 100
        val_loss, val_acc = evaluate(model, val_loader, device)

        tqdm.write(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.2f}% | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%"
        )

        completed_epoch = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model,
                optim,
                scheduler,
                completed_epoch,
                best_val_loss,
                model_config,
                tokenizer_config,
            )
            tqdm.write(f"Saved new best model (val loss: {best_val_loss:.4f})")

        if completed_epoch % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{completed_epoch:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optim,
                scheduler,
                completed_epoch,
                best_val_loss,
                model_config,
                tokenizer_config,
            )
            tqdm.write(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
