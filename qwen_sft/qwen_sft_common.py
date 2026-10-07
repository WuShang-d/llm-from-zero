"""Shared local data, model, and generation helpers for the Qwen SFT lesson."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE / "models/Qwen3-0.6B-Base"
SFT_DIR = HERE / "final_outputs/best_model"
TEST_FILE = HERE / "data/test.jsonl"


def read_test_rows(path: Path):
    if not path.is_file():
        raise ValueError(f"Test file does not exist: {path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                prompt, completion = row["prompt"], row["completion"]
                if not isinstance(prompt, list) or not prompt or not isinstance(completion, list) or not completion:
                    raise ValueError("prompt and completion must be nonempty lists")
                if prompt[-1]["role"] != "user" or completion[0]["role"] != "assistant":
                    raise ValueError("expected a user prompt and an assistant completion")
                rows.append(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid test example at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"Test file is empty: {path}")
    return rows


def load_tokenizer(model_dir: Path):
    if not model_dir.is_dir():
        raise ValueError(f"Model/tokenizer directory does not exist: {model_dir}")
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def load_model(model_dir: Path):
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return model.to(device).eval()


def generate(model, tokenizer, messages, max_new_tokens: int,
             repetition_penalty: float = 1.1) -> str:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False,  # Match the empty think block inserted in SFT examples.
        return_tensors="pt", return_dict=True,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    stop_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
    stop_ids = list(dict.fromkeys(value for value in stop_ids if value is not None))
    with torch.inference_mode():
        result = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = result[0, encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
