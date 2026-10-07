"""Clean two local AI/Agent QA sources into this project's chat SFT format.

Run from the project root:
    uv run python data/prepare_ai_qa_sft.py

This script checks structure and token length, not factual correctness. Review the
accepted rows and their source URLs before treating them as trusted supervision.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer.bpe import BPEEncoder, get_merges  # noqa: E402


DATA_DIR = ROOT / "qwen_sft" / "data" / "legacy_ai_qa"
SOURCES = (
    ("ai_agents_en", DATA_DIR / "raw" / "dataset_en.jsonl", "MIT"),
    ("qa_ml_dl_jsonl", DATA_DIR / "raw" / "brief_questions_with_answers.jsonl", "MIT"),
)
CHECKPOINT = ROOT / "checkpoints" / "fineweb_100m" / "best_model.pt"
MERGES_FILE = ROOT / "tokenizer" / "fineweb_100m_merges.txt"

# The larger source covers many unrelated ML topics. Keep LLM foundations and
# explicit LLM/Agent questions; do not interpret every use of "transformer" or
# "agent" as a language-model question.
LLM_TOPICS = {
    "attention mechanisms", "fine tuning", "hugging face transformers library",
    "language modeling", "language models", "model fine tuning",
    "multi head attention", "original transformer", "self attention",
    "tokenization", "transformer architecture", "transformer models",
}
LLM_QUESTION = re.compile(
    r"\b(?:llms?|large language models?|language models?|tokeniz\w*|"
    r"self.attention|multi.head attention|prompt engineering|"
    r"retrieval.augmented generation|\brag\b|generative ai|"
    r"chatbots?|ai agents?)\b",
    re.IGNORECASE,
)


def normalize_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def question_key(question):
    return re.sub(r"[^\w]+", " ", question.casefold()).strip()


def topic_key(topic):
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(topic).casefold()).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(source_name, path, license_name, counts):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            counts[f"{source_name}:rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts[f"{source_name}:invalid_json"] += 1
                continue
            if not isinstance(row, dict):
                counts[f"{source_name}:invalid_record"] += 1
                continue
            if source_name == "ai_agents_en":
                if row.get("language") != "en":
                    counts[f"{source_name}:non_english"] += 1
                    continue
            else:
                topic = topic_key(row.get("topic", ""))
                if topic not in LLM_TOPICS and not LLM_QUESTION.search(
                    normalize_text(row.get("question"))
                ):
                    counts[f"{source_name}:outside_scope"] += 1
                    continue
            question = normalize_text(row.get("question"))
            answer = normalize_text(row.get("answer"))
            if not question or not answer or "<|" in question or "<|" in answer:
                counts[f"{source_name}:invalid_content"] += 1
                continue
            if len(question) < 8 or len(answer) < 12:
                counts[f"{source_name}:too_short"] += 1
                continue
            yield {
                "source": source_name,
                "source_id": str(row.get("id") or line_number),
                "license": license_name,
                "topic": normalize_text(row.get("topic") or row.get("category")),
                "source_url": normalize_text(row.get("source_url")),
                "question": question,
                "answer": answer,
            }


def token_length(row, encoder, special_ids):
    # Mirrors chat_data.encode_conversations() for a single user/assistant pair.
    full = (
        [special_ids["<|bos|>"]]
        + encoder.encode(f"User: {row['question']}\n")
        + encoder.encode("Assistant: ")
        + encoder.encode(row["answer"])
        + [special_ids["<|eos|>"]]
    )
    return len(full) - 1


def split_rows(rows, seed):
    # Split separately by source. Exact normalized questions have already been
    # grouped, so no identical question can cross train/valid/test.
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    splits = {"train": [], "valid": [], "test": []}
    for source_name, group in sorted(by_source.items()):
        random.Random(f"{seed}:{source_name}").shuffle(group)
        n = len(group)
        n_test = max(1, round(n * 0.1)) if n >= 10 else 0
        n_valid = max(1, round(n * 0.1)) if n >= 10 else 0
        splits["test"].extend(group[:n_test])
        splits["valid"].extend(group[n_test:n_test + n_valid])
        splits["train"].extend(group[n_test + n_valid:])
    return splits


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    for _, path, _ in SOURCES:
        if not path.is_file():
            parser.error(f"Missing raw input: {path}")
    if not CHECKPOINT.is_file() or not MERGES_FILE.is_file():
        parser.error("The FineWeb 100M checkpoint or matching merges file is missing")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint["model_config"]
    tokenizer = checkpoint["tokenizer_config"]
    if get_merges(MERGES_FILE) != tokenizer["merges"]:
        parser.error("The named merges file does not match the checkpoint")
    encoder = BPEEncoder(tokenizer["merges"], tokenizer["special_ids"])
    max_length = config["block_size"]

    counts = Counter()
    grouped = defaultdict(list)
    for source_name, path, license_name in SOURCES:
        for row in read_rows(source_name, path, license_name, counts):
            length = token_length(row, encoder, tokenizer["special_ids"])
            if length > max_length:
                counts[f"{source_name}:over_context"] += 1
                continue
            row["input_tokens"] = length
            grouped[question_key(row["question"])].append(row)

    cleaned = []
    for group in grouped.values():
        distinct_answers = {normalize_text(r["answer"]).casefold() for r in group}
        if len(distinct_answers) != 1:
            counts["conflicting_duplicate_questions"] += len(group)
            continue
        cleaned.append(group[0])
        counts["exact_duplicate_questions_removed"] += len(group) - 1

    splits = split_rows(cleaned, args.seed)
    if not splits["train"] or not splits["valid"] or not splits["test"]:
        parser.error("Filtering left an empty split; review scope rules and raw data")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for split, rows in splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", [
            {"messages": [
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["answer"]},
            ]} for r in rows
        ])
        audit.extend({"split": split, **r} for r in rows)
    write_jsonl(args.output_dir / "provenance.jsonl", audit)
    manifest = {
        "seed": args.seed,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "max_input_tokens": max_length,
        "tokenizer_merges": str(MERGES_FILE.relative_to(ROOT)),
        "tokenizer_merges_sha256": sha256(MERGES_FILE),
        "raw_sources": [
            {"name": name, "path": str(path.relative_to(ROOT)),
             "sha256": sha256(path), "license": license_name}
            for name, path, license_name in SOURCES
        ],
        "counts": dict(sorted(counts.items())),
        "splits": {name: {"rows": len(rows),
                          "sources": dict(Counter(r["source"] for r in rows))}
                   for name, rows in splits.items()},
        "review_status": "Structural cleaning only; answers and source claims need human fact review.",
        "split_note": "Exact normalized questions kept in one split; semantic paraphrases are not detected.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": manifest["counts"], "splits": manifest["splits"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
