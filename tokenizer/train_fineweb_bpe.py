"""Train a separate BPE vocabulary from the FineWeb-Edu training split.

Run from the repository root with:
    python -m tokenizer.train_fineweb_bpe
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizer.bpe import (
    BPEEncoder,
    assign_special_ids,
    build_vocab,
    save_merges,
    save_vocab,
    train_bpe,
)


ROOT = Path(__file__).resolve().parent.parent
SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>"]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data" / "fineweb_edu" / "train.txt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "tokenizer" / "fineweb_edu_8k")
    parser.add_argument("--sample-chars", type=positive_int, default=50_000_000)
    parser.add_argument("--vocab-size", type=positive_int, default=8192)
    return parser.parse_args()


def train(source: Path, output_dir: Path, sample_chars: int, vocab_size: int) -> dict:
    if vocab_size < 256 + len(SPECIAL_TOKENS) or vocab_size > 65535:
        raise ValueError("vocab_size must be between 259 and 65535 for uint16 caches")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; choose a new output directory")

    # The downloaded documents were shuffled before being written. Training
    # on a bounded prefix avoids loading the full multi-GB corpus into RAM.
    with source.open("r", encoding="utf-8") as file:
        source_sample = file.read(sample_chars)
    if not source_sample:
        raise ValueError("Training source is empty")
    sample = source_sample.replace("<|endoftext|>", "<|eos|>")
    sample_hash = hashlib.sha256(sample.encode("utf-8")).hexdigest()

    merges = train_bpe(sample, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
    special_ids = assign_special_ids(merges, SPECIAL_TOKENS)
    vocab = build_vocab(merges, special_ids)
    if len(vocab) != vocab_size:
        raise RuntimeError(f"BPE produced {len(vocab)} tokens, expected {vocab_size}")

    encoder = BPEEncoder(merges, special_ids)
    check_text = "Hello, world! The rain made the garden green."
    encoded = encoder.encode(check_text)
    decoded = b"".join(vocab[token] for token in encoded).decode("utf-8")
    if decoded != check_text:
        raise RuntimeError("Tokenizer round-trip check failed")

    output_dir.mkdir(parents=True)
    save_merges(merges, output_dir / "merges.txt")
    save_vocab(vocab, special_ids, output_dir / "vocab.json")
    try:
        source_name = str(source.resolve().relative_to(ROOT))
    except ValueError:
        source_name = str(source.resolve())
    metadata = {
        "source": source_name,
        "source_size_bytes": source.stat().st_size,
        "sample_method": "first characters of document-shuffled training split",
        "source_sample_chars": len(source_sample),
        "training_sample_chars": len(sample),
        "sample_sha256": sample_hash,
        "vocab_size": vocab_size,
        "merge_count": len(merges),
        "special_ids": special_ids,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    args = parse_args()
    metadata = train(args.source, args.output_dir, args.sample_chars, args.vocab_size)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
