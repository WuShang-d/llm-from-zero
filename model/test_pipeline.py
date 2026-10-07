"""Behavioral checks for attention, chat labels and checkpoint loading."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from chat_data import collate_chat, encode_conversations
from chat_sft import supervised_loss
from generate import generate, load_model
from model.attention import Attention
from model.rope import apply_rope
from model.transformer import TransformerLM
from tokenizer.bpe import BPEEncoder, build_vocab


SPECIAL_IDS = {"<|bos|>": 256, "<|eos|>": 257, "<|pad|>": 258}


class PipelineTest(unittest.TestCase):
    def test_attention_cannot_see_future_tokens(self):
        torch.manual_seed(0)
        attention = Attention(num_heads=2, d_model=8).eval()
        tokens = torch.randn(1, 4, 8)
        changed = tokens.clone()
        changed[:, 2:] += 100
        with torch.no_grad():
            original = attention(tokens)
            modified = attention(changed)
        torch.testing.assert_close(original[:, :2], modified[:, :2])
        self.assertFalse(torch.allclose(original[:, 2:], modified[:, 2:]))

    def test_rope_preserves_norm_and_rotates_by_position(self):
        q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        rotated, _ = apply_rope(q, q)
        torch.testing.assert_close(rotated[..., 0, :], q[..., 0, :])
        torch.testing.assert_close(rotated[..., 1, :],
                                   torch.tensor([[[torch.cos(torch.tensor(1.0)),
                                                   torch.sin(torch.tensor(1.0))]]]))
        torch.testing.assert_close(rotated.norm(dim=-1), q.norm(dim=-1))

    def test_chat_loss_only_covers_answer_and_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.jsonl"
            path.write_text(json.dumps({"messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]}) + "\n", encoding="utf-8")
            sample = encode_conversations(path, [], SPECIAL_IDS, 128)[0]

        inputs, labels = sample
        answer = BPEEncoder([], SPECIAL_IDS).encode("Hello") + [SPECIAL_IDS["<|eos|>"]]
        self.assertEqual([x for x in labels if x != -100], answer)
        self.assertEqual(labels[-len(answer):], answer)
        self.assertTrue(all(x == -100 for x in labels[:-len(answer)]))
        padded_inputs, padded_labels = collate_chat([sample, ([1], [-100])], SPECIAL_IDS["<|pad|>"])
        self.assertEqual(padded_inputs.shape, padded_labels.shape)
        self.assertTrue(torch.all(padded_labels[1] == -100))

        class FixedLogits(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.values = values

            def forward(self, _):
                return self.values

        logits = torch.randn(1, len(inputs), 259)
        target = torch.tensor([labels])
        first = supervised_loss(FixedLogits(logits), torch.tensor([inputs]), target)
        logits[:, :len(inputs) - len(answer)] += 1000
        second = supervised_loss(FixedLogits(logits), torch.tensor([inputs]), target)
        torch.testing.assert_close(first, second)

    def test_checkpoint_reload_keeps_greedy_generation(self):
        torch.manual_seed(1)
        model = TransformerLM(vocab_size=259, num_heads=2, d_model=8, num_layers=1).eval()
        config = {"vocab_size": 259, "num_heads": 2, "d_model": 8,
                  "num_layers": 1, "block_size": 8}
        tokenizer = {"merges": [], "special_ids": SPECIAL_IDS}
        vocab = build_vocab([], SPECIAL_IDS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            torch.save({"model_state_dict": model.state_dict(), "model_config": config,
                        "tokenizer_config": tokenizer}, path)
            loaded, loaded_vocab, merges, special_ids, block_size = load_model(path, "cpu")
            with torch.no_grad():
                expected = generate(model, "Hi", 3, vocab, [], SPECIAL_IDS, 8,
                                    "cpu", temperature=0)
                actual = generate(loaded, "Hi", 3, loaded_vocab, merges, special_ids,
                                  block_size, "cpu", temperature=0)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
