"""Evaluate completion loss and next-token accuracy on the held-out test set.

These are teacher-forced token metrics, not the percentage of factually correct
generated answers. Use compare_qwen_sft.py to inspect answer quality.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from qwen_sft_common import BASE_DIR, SFT_DIR, TEST_FILE, load_tokenizer, read_test_rows


def evaluate(model_dir: Path, rows: list, max_length: int, batch_size: int):
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    tokenizer = load_tokenizer(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    config = SFTConfig(
        output_dir=str(SFT_DIR.parent / "test_eval_tmp"),
        per_device_eval_batch_size=batch_size,
        max_length=max_length,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        dataloader_pin_memory=False,
        use_cpu=not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    )
    dataset = Dataset.from_list(rows)
    # TRL requires a train_dataset at construction; train() is never called.
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset, eval_dataset=dataset,
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate()
    del trainer, model
    return {
        "loss": metrics["eval_loss"],
        "perplexity": math.exp(metrics["eval_loss"]),
        "mean_token_accuracy": metrics.get("eval_mean_token_accuracy"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TEST_FILE)
    parser.add_argument("--sft-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--include-base", action="store_true", help="Evaluate Base on the same test set too")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=SFT_DIR.parent / "test_metrics.json")
    args = parser.parse_args()
    if args.max_length < 1 or args.batch_size < 1:
        parser.error("--max-length and --batch-size must be positive")

    rows = read_test_rows(args.data)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    results = {"test_file": str(args.data.resolve()), "test_sha256": digest,
               "examples": len(rows), "max_length": args.max_length,
               "metrics_note": "Teacher-forced completion metrics; token accuracy is not factual answer accuracy."}
    if args.include_base:
        results["base"] = evaluate(args.base_dir, rows, args.max_length, args.batch_size)
    results["sft"] = evaluate(args.sft_dir, rows, args.max_length, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
