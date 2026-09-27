"""Stream a bounded FineWeb-Edu-Dedup sample into separate text files.

The published token_count field uses the GPT-2 tokenizer. It is only a
download budget: recount with the tokenizer chosen for the new model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET = "HuggingFaceTB/smollm-corpus"
CONFIG = "fineweb-edu-dedup"
ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tokens", type=positive_int, default=2_000_000_000,
                        help="Approximate GPT-2 tokens to save for training (default: 2B)")
    parser.add_argument("--valid-tokens", type=positive_int, default=20_000_000,
                        help="Approximate GPT-2 tokens to save for validation (default: 20M)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fineweb_edu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=positive_int, default=10_000)
    parser.add_argument("--min-chars", type=positive_int, default=200)
    return parser.parse_args()


def is_validation(doc_id: str, seed: int) -> bool:
    """Stable document-level split, independent of streaming order."""
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100 == 0


def write_sample(rows, output_dir: Path, train_budget: int, valid_budget: int,
                 seed: int, min_chars: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {name: output_dir / f"{name}.txt" for name in ("train", "valid")}
    for path in final_paths.values():
        if path.exists():
            raise FileExistsError(f"{path} already exists; move it before starting a new sample")

    temp_paths = {name: path.with_suffix(".txt.part") for name, path in final_paths.items()}
    for path in temp_paths.values():
        if path.exists():
            raise FileExistsError(f"{path} exists from an interrupted run; inspect or remove it first")

    budgets = {"train": train_budget, "valid": valid_budget}
    counts = {"train": 0, "valid": 0}
    documents = {"train": 0, "valid": 0}
    seen = 0
    try:
        with (temp_paths["train"].open("w", encoding="utf-8") as train_file,
              temp_paths["valid"].open("w", encoding="utf-8") as valid_file):
            files = {"train": train_file, "valid": valid_file}
            for row in rows:
                seen += 1
                text = row.get("text") or ""
                if len(text) < min_chars:
                    continue
                doc_id = row.get("id")
                token_count = (row.get("metadata") or {}).get("token_count")
                if not doc_id or not isinstance(token_count, int) or token_count <= 0:
                    continue
                split = "valid" if is_validation(str(doc_id), seed) else "train"
                if counts[split] >= budgets[split] or counts[split] + token_count > budgets[split]:
                    continue
                # Keep documents intact and use the separator expected by ensure_token_cache.
                files[split].write(text.rstrip() + "\n<|endoftext|>\n")
                counts[split] += token_count
                documents[split] += 1
                if seen % 10_000 == 0:
                    print(f"Read {seen:,} docs | train {counts['train']:,}/{train_budget:,} | "
                          f"valid {counts['valid']:,}/{valid_budget:,}", flush=True)
                if all(counts[name] >= budgets[name] * 0.999 for name in budgets):
                    break
        if any(counts[name] < budgets[name] * 0.999 for name in budgets):
            raise RuntimeError(f"Stream ended before budgets were reached: {counts}")
        for name in budgets:
            temp_paths[name].replace(final_paths[name])
        manifest = {
            "source": DATASET,
            "config": CONFIG,
            "seed": seed,
            "split": "SHA-256(seed:document_id) mod 100 == 0 for validation",
            "token_count_note": "Source metadata uses GPT-2; recount with the new tokenizer",
            "approx_gpt2_tokens": counts,
            "documents": documents,
            "documents_read": seen,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return manifest
    except Exception:
        # Retain .part files for inspection; a fresh run must not overwrite them.
        raise


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install the Hugging Face datasets package: pip install datasets") from error

    stream = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    rows = iter(stream)
    try:
        manifest = write_sample(rows, args.output_dir, args.train_tokens,
                                args.valid_tokens, args.seed, args.min_chars)
    finally:
        # A bounded run stops mid-stream. Close the generator so its Parquet
        # readers release pending network requests before Python exits.
        rows.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
