"""Train the language model on the official TinyStories train/validation files."""

import argparse
import math
import time
from contextlib import nullcontext
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
grad_accum_steps = 1
amp = "auto"  # CUDA 上优先 bf16，否则 fp16；CPU/MPS 保持 fp32。
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="每次前向传播的 micro-batch 大小")
    parser.add_argument("--grad-accum-steps", type=int, default=grad_accum_steps,
                        help="每次参数更新累积的 micro-batch 数")
    parser.add_argument("--amp", choices=("auto", "off", "bf16", "fp16"),
                        default=amp)
    parser.add_argument("--max-steps", type=int,
                        help="最多执行多少次参数更新，用于短程试跑")
    parser.add_argument("--max-val-batches", type=int,
                        help="验证集最多运行多少个 batch，用于短程试跑")
    parser.add_argument("--no-save", action="store_true",
                        help="不写入检查点，用于短程试跑")
    args = parser.parse_args()
    for name in ("batch_size", "grad_accum_steps", "max_steps", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if device != "cuda" and args.amp in ("bf16", "fp16"):
        parser.error("当前 AMP 仅支持 CUDA；在 CPU/MPS 上请使用 --amp off/auto")
    # 短程试跑的验证结果不可与完整 epoch 比较，也不能覆盖正式检查点。
    args.no_save = args.no_save or args.max_steps is not None or args.max_val_batches is not None
    return args


def get_amp_dtype(mode):
    if device != "cuda" or mode == "off":
        return None
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 CUDA 设备不支持 bf16，请改用 fp16 或 auto")
    return torch.bfloat16 if mode == "bf16" else torch.float16


def synchronize_device():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    args = parse_args()
    amp_dtype = get_amp_dtype(args.amp)
    # torch.amp.GradScaler 在较新的 PyTorch 才提供；兼容 requirements 的 2.0 下限。
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    print(f"Train with: {device}")
    print(
        f"Micro-batch: {args.batch_size} | Accumulation: {args.grad_accum_steps} | "
        f"Effective batch: {args.batch_size * args.grad_accum_steps} | "
        f"AMP: {str(amp_dtype).replace('torch.', '') if amp_dtype else 'off'}"
    )
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
        args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        args.batch_size,
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
    total_steps = num_epochs * math.ceil(len(train_loader) / args.grad_accum_steps)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = create_cosine_scheduler(
        optim,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
    if not args.no_save:
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

    total_optimizer_steps = 0
    for epoch in range(num_epochs):
        train_loss_sum = 0.0
        train_correct = 0
        train_tokens = 0
        optimizer_steps = 0
        optim.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        synchronize_device()
        train_start = time.perf_counter()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )
        for batch_index, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)

            context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_dtype is not None else nullcontext()
            )
            with context:
                logits = model(x)  # [B, T, V]
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    y.reshape(B * T),
                )

            # 最后一组可能不足 grad_accum_steps，按实际 micro-batch 数归一化。
            group_start = (batch_index // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, len(train_loader) - group_start)
            scaler.scale(loss / group_size).backward()
            if (batch_index + 1) % args.grad_accum_steps == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_grad_norm,
                )
                previous_scale = scaler.get_scale()
                scaler.step(optim)
                scaler.update()
                # fp16 溢出时 GradScaler 会跳过 optimizer.step()。
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    optimizer_steps += 1
                    total_optimizer_steps += 1
                optim.zero_grad(set_to_none=True)

            with torch.no_grad():
                y_pred = logits.argmax(dim=-1)
                train_correct += (y_pred == y).sum().item()

            token_count = y.numel()
            batch_loss = loss.item()
            train_loss_sum += batch_loss * token_count
            train_tokens += token_count

            progress.set_postfix(
                loss=f"{batch_loss:.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}",
            )
            if args.max_steps is not None and total_optimizer_steps >= args.max_steps:
                break

        synchronize_device()
        train_seconds = time.perf_counter() - train_start
        peak_memory = (
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            if device == "cuda" else "N/A (CUDA only)"
        )

        train_loss = train_loss_sum / train_tokens
        train_acc = train_correct / train_tokens * 100
        val_loss, val_acc = evaluate(
            model, val_loader, device,
            amp_dtype=amp_dtype, max_batches=args.max_val_batches,
        )

        tqdm.write(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.2f}% | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%"
        )
        tqdm.write(
            f"Train speed: {train_tokens / train_seconds:,.0f} tokens/s | "
            f"train time: {train_seconds:.1f}s | "
            f"updates: {optimizer_steps:,} | peak CUDA memory: {peak_memory}"
        )

        completed_epoch = epoch + 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not args.no_save:
                save_checkpoint(
                    checkpoint_dir / "best_model.pt",
                    model, optim, scheduler, completed_epoch,
                    best_val_loss, model_config, tokenizer_config,
                )
                tqdm.write(f"Saved new best model (val loss: {best_val_loss:.4f})")

        if not args.no_save and completed_epoch % checkpoint_every == 0:
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

        if args.max_steps is not None and total_optimizer_steps >= args.max_steps:
            break

if __name__ == "__main__":
    main()
