"""A small conversation SFT lab built on this project's pretrained LM.

This is a teaching experiment, not a general-purpose assistant recipe.
"""

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chat_data import ChatDataset, collate_chat, encode_conversations, encode_prompt
from model.transformer import TransformerLM
from tokenizer.bpe import bpe_decode, build_vocab


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT / "checkpoints" / "best_model.pt"
DEFAULT_CHAT = ROOT / "checkpoints" / "chat_demo.pt"
DEFAULT_TRAIN = ROOT / "data" / "chat_demo_train.jsonl"
DEFAULT_VALID = ROOT / "data" / "chat_demo_valid.jsonl"


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    tokenizer = checkpoint["tokenizer_config"]
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        num_heads=config["num_heads"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), config, tokenizer


def select_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loader(examples, batch_size, pad_id, shuffle):
    return DataLoader(
        ChatDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_chat(batch, pad_id),
    )


def supervised_loss(model, input_ids, labels):
    logits = model(input_ids)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        count = (labels != -100).sum().item()
        weighted_loss += supervised_loss(model, inputs, labels).item() * count
        token_count += count
    model.train()
    return weighted_loss / token_count


def prepare_examples(path, tokenizer, max_length):
    return encode_conversations(
        path,
        tokenizer["merges"],
        tokenizer["special_ids"],
        max_length,
    )


def inspect(args):
    _, config, tokenizer = load_model(args.base, "cpu")
    examples = prepare_examples(args.data, tokenizer, config["block_size"])
    vocab = build_vocab(tokenizer["merges"], tokenizer["special_ids"])
    input_ids, labels = examples[0]
    answer_ids = [token for token in labels if token != -100]
    print(f"Examples: {len(examples)} | input tokens: {len(input_ids)} | answer tokens: {len(answer_ids)}")
    print("Model input:")
    print(bpe_decode(input_ids, vocab))
    print("Loss is computed only on:")
    print(bpe_decode(answer_ids, vocab))


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("steps, batch-size and eval-every must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    model, config, tokenizer = load_model(args.base, device)
    max_length = config["block_size"]
    train_examples = prepare_examples(args.train_data, tokenizer, max_length)
    valid_examples = prepare_examples(args.valid_data, tokenizer, max_length)
    pad_id = tokenizer["special_ids"]["<|pad|>"]
    train_loader = make_loader(train_examples, args.batch_size, pad_id, True)
    valid_loader = make_loader(valid_examples, args.batch_size, pad_id, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_iterator = iter(train_loader)
    best_valid = math.inf
    print(
        f"device={device} train_examples={len(train_examples)} "
        f"valid_examples={len(valid_examples)} max_length={max_length}"
    )
    model.train()

    for step in range(1, args.steps + 1):
        try:
            inputs, labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            inputs, labels = next(train_iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(model, inputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            valid = validation_loss(model, valid_loader, device)
            print(f"step={step} train_loss={loss.item():.4f} valid_loss={valid:.4f}")
            if valid < best_valid and not args.no_save:
                best_valid = valid
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": config,
                        "tokenizer_config": tokenizer,
                        "sft_step": step,
                        "valid_loss": valid,
                    },
                    args.output,
                )
                print(f"saved={args.output}")


@torch.no_grad()
def respond(model, prompt_ids, vocab, eos_id, max_new_tokens, temperature, device, context_size):
    model.eval()
    tokens = list(prompt_ids)
    answer = []
    for _ in range(max_new_tokens):
        context = torch.tensor([tokens[-context_size:]], dtype=torch.long, device=device)
        logits = model(context)[0, -1]
        if temperature == 0:
            next_id = int(logits.argmax())
        else:
            values, indices = torch.topk(logits / temperature, k=min(40, logits.numel()))
            choice = torch.multinomial(torch.softmax(values, dim=-1), 1)
            next_id = int(indices[choice])
        if next_id == eos_id:
            break
        answer.append(next_id)
        tokens.append(next_id)
    return bpe_decode(answer, vocab)


def chat(args):
    if args.max_new_tokens < 1 or args.temperature < 0:
        raise ValueError("max-new-tokens must be positive and temperature nonnegative")
    device = select_device(args.device)
    model, config, tokenizer = load_model(args.checkpoint, device)
    merges = tokenizer["merges"]
    special_ids = tokenizer["special_ids"]
    vocab = build_vocab(merges, special_ids)
    history = []

    while True:
        user_text = args.prompt if args.prompt is not None else input("User: ").strip()
        if not user_text:
            break
        prompt_ids = encode_prompt(user_text, history, merges, special_ids, config["block_size"])
        answer = respond(
            model, prompt_ids, vocab, special_ids["<|eos|>"],
            args.max_new_tokens, args.temperature, device, config["block_size"],
        )
        print(f"Assistant: {answer}")
        history.append((user_text, answer))
        if args.prompt is not None:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser("inspect", help="Show one encoded teaching example")
    inspect_parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    inspect_parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN)
    inspect_parser.set_defaults(run=inspect)

    train_parser = commands.add_parser("train", help="Fine-tune the 17M base model")
    train_parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    train_parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    train_parser.add_argument("--valid-data", type=Path, default=DEFAULT_VALID)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_CHAT)
    train_parser.add_argument("--steps", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--eval-every", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train_parser.add_argument("--no-save", action="store_true")
    train_parser.set_defaults(run=train)

    chat_parser = commands.add_parser("chat", help="Try the fine-tuned checkpoint")
    chat_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHAT)
    chat_parser.add_argument("--prompt", help="One-shot prompt; omit for interactive chat")
    chat_parser.add_argument("--max-new-tokens", type=int, default=64)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    chat_parser.set_defaults(run=chat)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
