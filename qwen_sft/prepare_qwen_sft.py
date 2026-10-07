"""Convert reviewed question/answer records to Qwen prompt-completion JSONL.

Default inputs are the previous SFT data and project-authored daily chat.
Both preserve their train/valid/test splits. Run from the project root:
    .venv/bin/python qwen_sft/prepare_qwen_sft.py
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
SPLITS = ("train", "valid", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def validate_record(record, line_number):
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")
    missing = {"split", "question", "answer"} - record.keys()
    if missing:
        raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
    split = record["split"]
    if split not in SPLITS:
        raise ValueError(f"line {line_number}: invalid split {split!r}")
    question, answer = record["question"], record["answer"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"line {line_number}: question must be nonempty text")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"line {line_number}: answer must be nonempty text")
    return split, question.strip(), answer.strip()


def validate_example(example, line_number):
    if set(example) != {"prompt", "completion"}:
        raise ValueError(f"line {line_number}: incorrect output fields")
    for field, role in (("prompt", "user"), ("completion", "assistant")):
        messages = example[field]
        if (not isinstance(messages, list) or len(messages) != 1 or
                not isinstance(messages[0], dict) or
                set(messages[0]) != {"role", "content"} or
                messages[0]["role"] != role or
                not isinstance(messages[0]["content"], str) or
                not messages[0]["content"].strip()):
            raise ValueError(f"line {line_number}: invalid {field} message or role")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=HERE / "data/legacy_ai_qa/provenance.jsonl",
                        help="Previous SFT JSONL with split, question and answer")
    parser.add_argument("--daily-input", type=Path,
                        default=HERE / "data/daily_chat.jsonl",
                        help="Project-authored daily conversation JSONL")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--model-dir", type=Path,
                        default=HERE / "models/Qwen3-0.6B-Base")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Maximum Qwen chat-template token count, including answer")
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    inputs = (args.input, args.daily_input)
    for path in inputs:
        if not path.is_file():
            parser.error(f"Input does not exist: {path}")
    if args.input.resolve() == args.daily_input.resolve():
        parser.error("--input and --daily-input must be different files")
    if not args.model_dir.is_dir():
        parser.error(f"Qwen model/tokenizer directory does not exist: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    output = {split: [] for split in SPLITS}
    seen = {}
    counts = Counter()
    source_counts = {path: Counter() for path in inputs}
    for path in inputs:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                location = f"{path}:{line_number}"
                if not line.strip():
                    raise ValueError(f"{location}: blank JSONL line")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{location}: invalid JSON: {error}") from error
                split, question, answer = validate_record(raw, location)
                key = question_key(question)
                if not key:
                    raise ValueError(f"{location}: question has no searchable characters")
                if key in seen:
                    previous_split, previous_answer = seen[key]
                    if previous_split != split:
                        raise ValueError(
                            f"{location}: duplicate question crosses "
                            f"{previous_split}/{split} splits: {question!r}")
                    if previous_answer != answer:
                        raise ValueError(f"{location}: duplicate question has conflicting answers")
                    counts["identical_duplicates_removed"] += 1
                    continue
                example = {
                    "prompt": [{"role": "user", "content": question}],
                    "completion": [{"role": "assistant", "content": answer}],
                }
                validate_example(example, location)
                messages = example["prompt"] + example["completion"]
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    enable_thinking=False)
                length = len(encoded["input_ids"])
                if length > args.max_length:
                    counts["over_length_removed"] += 1
                    continue
                seen[key] = (split, answer)
                output[split].append(example)
                source_counts[path][split] += 1

    if any(not rows for rows in output.values()):
        parser.error("At least one split is empty after filtering")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for example in rows:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
    manifest = {
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path),
                    "splits": dict(source_counts[path])} for path in inputs],
        "model_dir": str(args.model_dir.resolve()),
        "max_length": args.max_length,
        "length_method": "Qwen chat template, enable_thinking=False, full user+assistant turn",
        "splits": {split: {"rows": len(rows), "sha256": sha256(args.output_dir / f"{split}.jsonl")}
                   for split, rows in output.items()},
        "counts": dict(counts),
        "review_status": "Structure and length checked; answer accuracy requires human review.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"splits": manifest["splits"], "counts": manifest["counts"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
