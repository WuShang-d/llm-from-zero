"""Small, explicit conversation format for the chat SFT lesson."""

import json

import torch
from torch.utils.data import Dataset

from tokenizer.bpe import BPEEncoder


def encode_conversations(path, merges, special_ids, max_length):
    """Create one supervised sample per assistant turn.

    The model sees all earlier turns, but the loss covers only the current
    assistant answer and the existing EOS token. This keeps the pretrained
    8192-token vocabulary unchanged.
    """
    encoder = BPEEncoder(merges, special_ids)
    bos_id = special_ids["<|bos|>"]
    eos_id = special_ids["<|eos|>"]
    examples = []

    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
                raise ValueError(f"{path}:{line_number}: expected user/assistant pairs")

            context = [bos_id]
            for index, message in enumerate(messages):
                expected_role = "user" if index % 2 == 0 else "assistant"
                if message.get("role") != expected_role:
                    raise ValueError(f"{path}:{line_number}: expected {expected_role} at turn {index}")
                content = message.get("content")
                if not isinstance(content, str) or not content.strip() or "<|" in content:
                    raise ValueError(f"{path}:{line_number}: invalid content at turn {index}")

                if expected_role == "user":
                    context += encoder.encode(f"User: {content.strip()}\n")
                    continue

                prefix = encoder.encode("Assistant: ")
                answer = encoder.encode(content.strip())
                full = context + prefix + answer + [eos_id]
                if len(full) - 1 > max_length:
                    raise ValueError(
                        f"{path}:{line_number}: {len(full) - 1} input tokens exceed "
                        f"max_length={max_length}; shorten the example"
                    )
                target_start = len(context) + len(prefix)
                labels = [
                    token if position >= target_start else -100
                    for position, token in enumerate(full[1:], 1)
                ]
                examples.append((full[:-1], labels))
                context = full

    if not examples:
        raise ValueError(f"No assistant examples found in {path}")
    return examples


class ChatDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_chat(batch, pad_id):
    width = max(len(tokens) for tokens, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for row, (tokens, targets) in enumerate(batch):
        input_ids[row, :len(tokens)] = torch.tensor(tokens)
        labels[row, :len(targets)] = torch.tensor(targets)
    return input_ids, labels


def encode_prompt(user_text, history, merges, special_ids, max_length):
    """Use exactly the same segment boundaries as encode_conversations."""
    encoder = BPEEncoder(merges, special_ids)
    tokens = [special_ids["<|bos|>"]]
    for user, assistant in history:
        tokens += encoder.encode(f"User: {user}\n")
        tokens += encoder.encode("Assistant: ")
        tokens += encoder.encode(assistant)
        tokens.append(special_ids["<|eos|>"])
    tokens += encoder.encode(f"User: {user_text}\n")
    tokens += encoder.encode("Assistant: ")
    if len(tokens) > max_length:
        # Keep the newest user turn when a chat exceeds the 256-token model context.
        tokens = tokens[-max_length:]
    return tokens
