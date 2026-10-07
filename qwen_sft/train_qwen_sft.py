"""
用 `SFTTrainer` 加载 Qwen Base、它的 tokenizer、训练集和验证集。先设置小 batch、短序列、少量 `max_steps`，记录训练及验证 loss。代码还要保存：

- 模型和 tokenizer；
- 实际使用的训练配置；
- 数据文件的版本或校验值。

这一阶段才调用 `trainer.train()`。不要把仓库的 8192 词 BPE 或 `chat_sft.py` 里的模型类接进来；它们属于前几课的手写模型。
"""

import argparse
import json
import hashlib
from pathlib import Path
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = Path(__file__).resolve().parent

config = SFTConfig(
    output_dir=str(HERE / "final_outputs"),  # 保留前两轮的 outputs
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,            # 有效训练 batch = 8
    max_length=256,
    num_train_epochs=2,
    max_steps=-1,                             # 按 epoch 训练
    learning_rate=1e-5,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=36,
    save_strategy="steps",
    save_steps=36,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    completion_only_loss=True,
    packing=False,
    dataloader_pin_memory=False,              # 消除 MPS 提示
    seed=42,
    report_to="none",
)

def load_data(path: Path, kind: str):
    # kind: 数据，如 train, valid
    if not path.is_dir():
        raise ValueError(f"Data Dir does not exist: {path}")

    filename = path / f"{kind}.jsonl"
    if not filename.is_file():
        raise ValueError(f"Required data file does not exist: {filename}")

    data = []
    with filename.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {filename}:{line_number}") from exc
    return data

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=HERE / "data")
    parser.add_argument("--model-dir", type=Path,
                        default=HERE / "models/Qwen3-0.6B-Base")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error(f"Local model/tokenizer directory does not exist: {args.model_dir}")

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    train_data = load_data(args.data, "train")
    valid_data = load_data(args.data, "valid")

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_data),
        eval_dataset=Dataset.from_list(valid_data),
        processing_class=tokenizer,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "training_config.json").write_text(
        config.to_json_string(), encoding="utf-8"
    )
    (output_dir / "data_checksums.json").write_text(
        json.dumps(
            {
                "train.jsonl": sha256_file(args.data / "train.jsonl"),
                "valid.jsonl": sha256_file(args.data / "valid.jsonl"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer.train()

    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))       # 训练结束时已加载最佳 checkpoint
    tokenizer.save_pretrained(best_dir)

    (output_dir / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
