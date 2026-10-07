"""Inspect one local Qwen SFT example through TRL's preprocessing pipeline.

Loads only the local tokenizer and one local JSONL row. Does not load model
weights, download data, or start training.
"""

"""
这是本课最重要的代码。加载 `Qwen/Qwen3-0.6B-Base` 自带的 tokenizer，把一条样本交给准备使用的 TRL 数据处理流程，然后打印：

- 聊天模板格式化后的完整文本；
- 每个位置的 token ID 和可读 token；
- 每个位置的 label：`-100` 表示不计算 loss，其他值表示要预测该 token；
- 实际参与 loss 的文本，以及回答结束标记。

检查结果应当能说明：用户问题提供上下文，**回答 token 才是监督目标**。不要仅凭原始 JSON 的字段推断标签；要看处理后真正送进 batch 的 `labels`。TRL 文档说明，`prompt`/`completion` 数据默认只监督 completion；建议在配置里也显式写 `completion_only_loss=True`，并先关闭 `packing`，方便逐 token 核对。[TRL 文档](https://huggingface.co/docs/trl/sft_trainer)
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


HERE = Path(__file__).resolve().parent


def load_example(path: Path, index: int, prompt: str | None):
    if not path.is_file():
        raise ValueError(f"Data file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if prompt is not None:
                if row["prompt"][0]["content"].casefold() == prompt.casefold():
                    return row
            elif line_number - 1 == index:
                return row
    raise ValueError(f"No matching example in {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=HERE / "data/train.jsonl")
    parser.add_argument("--model-dir", type=Path,
                        default=HERE / "models/Qwen3-0.6B-Base")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--prompt", help="Find an exact user prompt, e.g. hi")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.index < 0 or args.max_length < 1:
        parser.error("--index must be nonnegative and --max-length positive")
    if not args.model_dir.is_dir():
        parser.error(f"Local model/tokenizer directory does not exist: {args.model_dir}")

    row = load_example(args.data, args.index, args.prompt)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    config = SFTConfig(max_length=args.max_length, completion_only_loss=True,
                       packing=False, report_to="none", use_cpu=True)

    # Use TRL's dataset preparation without constructing a model or trainer.
    context = SimpleNamespace(_tokenizer=tokenizer,
                              chat_template=tokenizer.chat_template,
                              completion_only_loss=config.completion_only_loss)
    prepared = SFTTrainer._prepare_dataset(
        context, Dataset.from_list([row]), tokenizer, config,
        packing=False, formatting_func=None, dataset_name="inspection")
    if len(prepared) != 1:
        raise RuntimeError("TRL removed the sample; check max_length")
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    batch = collator([prepared[0]])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    supervised = [token_id for token_id, label in zip(ids, labels) if label != -100]
    if not supervised:
        raise RuntimeError("No completion tokens are supervised")

    print(f"Data: {args.data}")
    print(f"Prompt: {row['prompt'][0]['content']}")
    print("Settings: completion_only_loss=True, packing=False, max_length="
          f"{args.max_length}")
    print("\nFormatted text:\n" + tokenizer.decode(ids, skip_special_tokens=False))
    print("\nposition\ttoken_id\ttoken\tlabel")
    for position, (token_id, label) in enumerate(zip(ids, labels)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{position}\t{token_id}\t{token!r}\t{label}")
    print("\nText contributing to loss:\n" +
          tokenizer.decode(supervised, skip_special_tokens=False))
    print("Final supervised token:", tokenizer.convert_ids_to_tokens(supervised[-1]))
    print("Masked prompt tokens:", sum(label == -100 for label in labels))
    print("Supervised completion tokens:", len(supervised))


if __name__ == "__main__":
    main()
