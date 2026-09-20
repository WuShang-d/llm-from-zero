"""
    运行：python3 -m unittest discover -s tokenizer -v

    核心不变量：bpe_decode(bpe_encode(text)) == text
"""

import tempfile
import unittest
from pathlib import Path

import bpe

SPECIALS = ["<|bos|>", "<|eos|>", "<|pad|>"]
VOCAB_SIZE = 400

TEXTS = [
    "",
    "a",
    "Hello, world!",
    "中国",
    "你好，世界！今天天气不错。",
    "🤣🤣🤣",
    "Hello World\n!",
    "  \n\n\t x  \r\n y",
    "_ 2024_x 'll 's",
    "é ² 　 \x00",
    "aaaa" * 50,
    "ab" * 101,
]


def train(corpus: str, specials: list[str]):
    merges = bpe.train_bpe(corpus, VOCAB_SIZE, specials)
    special_ids = bpe.assign_special_ids(merges, specials)
    vocab = bpe.build_vocab(merges, special_ids)
    return merges, special_ids, vocab


class BPETestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = (Path(__file__).parent / "corpus.txt").read_text(encoding="utf-8")[:100_000]
        cls.corpus = corpus
        cls.merges, cls.special_ids, cls.vocab = train(corpus, SPECIALS)

        # 只见过 ASCII 的分词器，用来测训练语料里没出现过的字符
        ascii_corpus = "the quick brown fox jumps over the lazy dog\n" * 200
        cls.ascii_merges, cls.ascii_special_ids, cls.ascii_vocab = train(ascii_corpus, SPECIALS)

    def roundtrip(self, text, merges=None, special_ids=None, vocab=None):
        merges = merges if merges is not None else self.merges
        special_ids = special_ids if special_ids is not None else self.special_ids
        vocab = vocab if vocab is not None else self.vocab
        ids = bpe.bpe_encode(text, merges, special_ids)
        self.assertEqual(bpe.bpe_decode(ids, vocab), text)
        return ids


class TestRoundtrip(BPETestCase):
    def test_edge_cases(self):
        for text in TEXTS:
            with self.subTest(text=text):
                self.roundtrip(text)

    def test_empty_and_single_char(self):
        self.assertEqual(bpe.bpe_encode("", self.merges), [])
        self.assertEqual(len(self.roundtrip("a")), 1)

    def test_unseen_characters(self):
        for text in ["中国", "🤣🤣🤣", "é\n\t", "the fox 你好 🤣"]:
            with self.subTest(text=text):
                self.roundtrip(text, self.ascii_merges, self.ascii_special_ids, self.ascii_vocab)

    def test_training_corpus_slice(self):
        self.roundtrip(self.corpus[:5000])

    def test_random_text(self):
        import random
        rng = random.Random(0)
        alphabet = list("abc XYZ_09'.,\n\t") + ["中", "é", "🤣", "́"]
        for _ in range(300):
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
            self.roundtrip(text)


class TestPretokenization(BPETestCase):
    def test_regex_covers_every_character(self):
        for text in TEXTS + [self.corpus[:5000]]:
            self.assertEqual("".join(bpe.PRETOKEN_PATTERN.findall(text)), text)

    def test_no_merge_across_pieces(self):
        for a, b in [("hello", "\nworld"), ("the", " cat"), ("end.", "\n\nnext")]:
            with self.subTest(a=a, b=b):
                joined = bpe.bpe_encode(a + b, self.merges)
                self.assertEqual(joined, bpe.bpe_encode(a, self.merges) + bpe.bpe_encode(b, self.merges))

    def test_encode_matches_sequential_merges(self):
        def sequential(text):
            out = []
            for piece in bpe.PRETOKEN_PATTERN.findall(text):
                tokens = bpe.byte_encode(piece)
                for pair, newid in self.merges:
                    tokens = bpe.update_tokens(tokens, pair, newid)
                out += tokens
            return out

        for text in TEXTS + [self.corpus[:3000]]:
            with self.subTest(text=text[:20]):
                self.assertEqual(bpe.bpe_encode(text, self.merges), sequential(text))


class TestVocab(BPETestCase):
    def test_size_includes_special_tokens(self):
        self.assertEqual(len(self.vocab), VOCAB_SIZE)
        self.assertEqual(sorted(self.vocab), list(range(VOCAB_SIZE)))

    def test_merge_bytes_are_concatenation(self):
        for (a, b), newid in self.merges:
            self.assertEqual(self.vocab[newid], self.vocab[a] + self.vocab[b])

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab_path, merges_path = Path(tmp) / "vocab.json", Path(tmp) / "merges.txt"
            bpe.save_vocab(self.vocab, self.special_ids, vocab_path)
            bpe.save_merges(self.merges, merges_path)
            vocab, special_ids = bpe.load_vocab(vocab_path)
            merges = bpe.get_merges(merges_path)
        self.assertEqual(vocab, self.vocab)
        self.assertEqual(special_ids, self.special_ids)
        self.assertEqual(merges, self.merges)

    def test_decode_incomplete_utf8_does_not_raise(self):
        self.assertEqual(bpe.bpe_decode([0xE4], self.vocab), "�")

    def test_vocab_size_too_small(self):
        with self.assertRaises(ValueError):
            bpe.train_bpe("abc", 257, SPECIALS)

    def test_empty_corpus(self):
        self.assertEqual(bpe.train_bpe("", 300, SPECIALS), [])


class TestSpecialTokens(BPETestCase):
    def test_ids_follow_merges_and_are_distinct(self):
        base = 256 + len(self.merges)
        self.assertEqual(self.special_ids, {t: base + k for k, t in enumerate(SPECIALS)})

    def test_encodes_to_single_id(self):
        for token, token_id in self.special_ids.items():
            self.assertEqual(bpe.bpe_encode(token, self.merges, self.special_ids), [token_id])

    def test_roundtrip_with_specials(self):
        for text in [
            "<|bos|>Hello<|eos|><|pad|>",
            "<|eos|>",
            "<|eos|><|eos|>",
            "text<|eos|>",
            "<|bos|>text",
            "he<|eos|>llo 中国\n<|pad|>🤣",
        ]:
            with self.subTest(text=text):
                self.roundtrip(text)

    def test_special_splits_words(self):
        eos = self.special_ids["<|eos|>"]
        ids = bpe.bpe_encode("he<|eos|>llo", self.merges, self.special_ids)
        self.assertEqual(ids, bpe.bpe_encode("he", self.merges) + [eos] + bpe.bpe_encode("llo", self.merges))

    def test_ordinary_text_never_yields_special_ids(self):
        ids = bpe.bpe_encode(self.corpus[:20000], self.merges, self.special_ids)
        self.assertLess(max(ids), 256 + len(self.merges))

    def test_without_special_ids_literal_is_plain_text(self):
        ids = bpe.bpe_encode("<|eos|>", self.merges)
        self.assertNotIn(self.special_ids["<|eos|>"], ids)
        self.assertEqual(bpe.bpe_decode(ids, self.vocab), "<|eos|>")

    def test_longer_special_wins_over_prefix(self):
        specials = ["<|a|>", "<|a|>b"]
        merges = bpe.train_bpe("hello world " * 20, 300, specials)
        special_ids = bpe.assign_special_ids(merges, specials)
        ids = bpe.bpe_encode("<|a|>b", merges, special_ids)
        self.assertEqual(ids, [special_ids["<|a|>b"]])

    def test_training_ignores_specials(self):
        corpus = "hello world <|eos|> " * 200
        merges = bpe.train_bpe(corpus, 300, ["<|eos|>"])
        vocab = bpe.build_vocab(merges, bpe.assign_special_ids(merges, ["<|eos|>"]))
        for _, newid in merges:
            self.assertNotIn(b"<", vocab[newid])
            self.assertNotIn(b"|", vocab[newid])

    def test_invalid_special_definitions(self):
        with self.assertRaises(ValueError):
            bpe.assign_special_ids(self.merges, ["<|a|>", "<|a|>"])
        with self.assertRaises(ValueError):
            bpe.assign_special_ids(self.merges, [""])


if __name__ == "__main__":
    unittest.main()
