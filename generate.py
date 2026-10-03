"""Generate text from a trained language-model checkpoint."""

import argparse
from pathlib import Path

import torch

from model.transformer import TransformerLM
from tokenizer.bpe import bpe_decode, bpe_encode, build_vocab


PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINTS = {
    "fineweb100m": PROJECT_DIR / "checkpoints" / "fineweb_100m" / "best_model.pt",
    "tinystories": PROJECT_DIR / "checkpoints" / "best_model.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="用预训练语言模型续写文本")
    parser.add_argument("--profile", choices=CHECKPOINTS, default="fineweb100m")
    parser.add_argument("--checkpoint", type=Path, help="覆盖所选配置的检查点路径")
    parser.add_argument("--prompt", help="只生成一次；不提供时进入交互模式")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 表示贪心解码")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, help="固定采样随机种子")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens 必须大于 0")
    if args.temperature < 0:
        parser.error("--temperature 不能小于 0")
    if args.top_k < 1:
        parser.error("--top-k 必须大于 0")
    return args


def select_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到检查点：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    tokenizer_config = checkpoint["tokenizer_config"]
    merges = tokenizer_config["merges"]
    special_ids = tokenizer_config["special_ids"]
    vocab = build_vocab(merges, special_ids)
    if len(vocab) != model_config["vocab_size"]:
        raise ValueError("检查点中的词表大小与模型配置不一致")

    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    block_size = model_config.get("block_size", 256)
    return model, vocab, merges, special_ids, block_size


@torch.inference_mode()
def generate(
    model,
    prompt,
    max_tokens,
    vocab,
    merges,
    special_ids,
    block_size,
    device,
    temperature=0.8,
    top_k=50,
):
    token_ids = bpe_encode(prompt, merges, special_ids)
    if not token_ids:
        raise ValueError("Prompt 不能为空")

    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    eos_id = special_ids.get("<|eos|>")
    top_k = min(top_k, len(vocab))

    for _ in range(max_tokens):
        # RoPE positions are relative to this window, matching training context.
        logits = model(tokens[:, -block_size:])[:, -1, :]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            values, indices = torch.topk(logits / temperature, k=top_k, dim=-1)
            sampled = torch.multinomial(torch.softmax(values, dim=-1), 1)
            next_token = indices.gather(-1, sampled)
        if eos_id is not None and next_token.item() == eos_id:
            break
        tokens = torch.cat((tokens, next_token), dim=1)

    return bpe_decode(tokens[0].tolist(), vocab)


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = select_device(args.device)
    checkpoint_path = args.checkpoint or CHECKPOINTS[args.profile]
    model, vocab, merges, special_ids, block_size = load_model(
        checkpoint_path, device
    )
    print(f"Loaded {checkpoint_path} on {device} (context: {block_size} tokens)")
    if args.profile == "fineweb100m":
        print("这是基础续写模型，尚未经过对话 SFT；建议输入英文续写开头。")

    def run(prompt):
        output = generate(
            model, prompt, args.max_new_tokens, vocab, merges, special_ids,
            block_size, device, args.temperature, args.top_k,
        )
        print(f"\n{output}\n")

    if args.prompt is not None:
        run(args.prompt)
        return

    while True:
        try:
            prompt = input("Prompt (/exit 退出)> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip() == "/exit":
            break
        if prompt:
            run(prompt)


if __name__ == "__main__":
    main()
