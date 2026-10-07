"""Chat interactively with the locally fine-tuned Qwen model."""

import argparse
from pathlib import Path

from qwen_sft_common import SFT_DIR, generate, load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--multi-turn", action="store_true", help="Keep conversation history between questions")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.repetition_penalty <= 0:
        parser.error("--max-new-tokens and --repetition-penalty must be positive")

    tokenizer = load_tokenizer(args.model_dir)
    model = load_model(args.model_dir)
    messages = []
    print("Qwen SFT is ready. Enter /reset to clear context or /exit to quit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question == "/exit":
            break
        if question == "/reset":
            messages.clear()
            print("Context cleared.")
            continue
        if not question:
            continue
        prompt = messages.copy() if args.multi_turn else []
        prompt.append({"role": "user", "content": question})
        answer = generate(model, tokenizer, prompt, args.max_new_tokens,
                          args.repetition_penalty)
        print(f"Qwen SFT: {answer}\n")
        if args.multi_turn:
            messages = prompt + [{"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
