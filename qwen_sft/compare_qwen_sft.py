"""Generate Base and SFT answers for the same held-out test prompts."""

import argparse
import json
from pathlib import Path

from qwen_sft_common import BASE_DIR, SFT_DIR, TEST_FILE, generate, load_model, load_tokenizer, read_test_rows


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TEST_FILE)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--limit", type=int, default=10, help="First N fixed test questions; 0 means all")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--output", type=Path, default=SFT_DIR.parent / "test_comparison.md")
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens < 1 or args.repetition_penalty <= 0:
        parser.error("--limit must be nonnegative; token limit and repetition penalty must be positive")

    rows = read_test_rows(args.data)
    if args.limit:
        rows = rows[:args.limit]
    tokenizer = load_tokenizer(args.base_dir)  # Exactly the same template and tokens on both sides.
    answers = []
    for name, model_dir in (("base", args.base_dir), ("sft", args.sft_dir)):
        print(f"Generating {name} answers for {len(rows)} test prompts...", flush=True)
        model = load_model(model_dir)
        answers.append([generate(model, tokenizer, row["prompt"], args.max_new_tokens,
                                 args.repetition_penalty) for row in rows])
        del model

    lines = ["# Qwen Base 与 SFT 固定测试题对比", "",
             f"测试集：`{args.data}`；题数：{len(rows)}；greedy 解码；enable_thinking=False；"
             f"max_new_tokens={args.max_new_tokens}；repetition_penalty={args.repetition_penalty}。",
             "两边使用 Base tokenizer 的同一个聊天模板和完全相同的输入。参考回答仅供人工核查。", "",
             "| # | 测试题 | 参考回答 | Base | SFT |", "|---:|---|---|---|---|"]
    records = []
    for index, (row, base, sft) in enumerate(zip(rows, *answers), 1):
        question = row["prompt"][-1]["content"]
        reference = row["completion"][0]["content"]
        records.append({"index": index, "prompt": row["prompt"], "reference": reference,
                        "base": base, "sft": sft})
        lines.append(f"| {index} | {markdown_cell(question)} | {markdown_cell(reference)} | "
                     f"{markdown_cell(base)} | {markdown_cell(sft)} |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved comparison: {args.output}\nSaved raw answers: {json_path}")


if __name__ == "__main__":
    main()
