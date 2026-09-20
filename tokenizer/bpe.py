"""
    BPE trainer
    
    train_bpe(corpus, vocab_size) -> 
    vocab.json, merges.txt
    
    - BPE trainer：读取训练文本，学习哪些 token 应该合并，产出合并规则
    - BPE tokenizer 的 encode：使用已训练好的合并规则，把新文本转 token id
    - BPE tokenizer 的 decode：把 token id 展开为 bytes，再还原文本
    训练只做一次；编码和解码会执行很多次
    
    训练需要设置目标词表大小，一般设为 1000
    基础词表：0 ～ 255，下一个可分配 token id 为256
    
    训练循环是：
    1. 准备训练语料的 byte token 序列。
    2. 统计所有相邻 pair 出现次数。
    3. 找频率最高的 pair。
    4. 分配一个新的 token id。
    5. 在训练语料中，将这个 pair 的所有非重叠出现位置替换为新 id。
    6. 保存这条 merge 规则。
    7. 重复以上过程。
    8. 当词表大小到达目标值，或者再也没有可合并 pair 时停止。
    如果目标词表为 1,000，理论上最多要学习：
    1,000 - 256 = 744 条 merge 规则
    每学到一条规则，就多一个 token。
    
    训练语料应如何组织
    最简单的做法：把整个训练文件 UTF-8 编码成一个很长的 byte 序列，直接训练。
    优点是简单，但它会允许跨边界合并。
    更常见的设计是：先把语料切成许多独立片段，再对每个片段训练。例如按：文档、行、段落、正则预分词得到的片段、单词和标点附近的片段。切开
    
    可以先用“按行切分”或“整份文本作为一个片段”，先跑通；随后再增加更合理的预分词。
    
    还应考虑：如果最高频 pair 的频率只有 1，要不要继续合并？
    答案取决于你的目标：
    - 教学/实验项目：通常仍然继续，直到目标词表大小；
    - 更实用的训练器：可以设最小频率阈值，例如只合并出现至少若干次的 pair，避免把偶然出现的片段塞进词表。
    
    建议的实现顺序
    可以按这个顺序自己写，每一步都验证后再进入下一步：
    1. 完成最基础的 byte encode 与 decode，确保任意 Unicode 文本可以往返恢复。
    2. 写“统计一条 token 序列中所有相邻 pair”的逻辑。
    3. 写“将指定 pair 从左到右非重叠替换为新 id”的逻辑。
    4. 用很小的手工数字序列验证合并结果。
    5. 把两者放进训练循环，生成 merge 表。
    6. 维护 token id 到 bytes 的 vocab。
    7. 用 merge 表实现 BPE encode。
    8. 实现能处理合并 token 的 decode。
    9. 验证训练语料和未见文本上的往返不变性。
    10. 最后再考虑预分词、特殊 token、最小频率和性能优化。
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

Pair = tuple[int, int]
Merge = tuple[Pair, int]
Vocab = dict[int, bytes]  # token id -> 原始 bytes
SpecialIds = dict[str, int]  # 特殊 token 字符串 -> token id

ROOT = Path(__file__).resolve().parent.parent
MERGES_PATH = ROOT / "merges.txt"
VOCAB_PATH = ROOT / "vocab.json"

# GPT-2 风格预分词（标准库 re 版本）：缩写、词（可带前导空格）、数字、标点、空白
PRETOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+"
)

def byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))

def byte_decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")

def split_special(text: str, special_tokens: Iterable[str]) -> list[str]:
    """
    按特殊 token 切分文本，返回 [普通, 特殊, 普通, 特殊, ..., 普通]：
    偶数下标是普通文本（可能为空串），奇数下标是特殊 token 本身。
    候选按长度倒序，保证 "<|a|>" 与 "<|a|>b" 同时存在时优先匹配更长的。
    """
    specials = sorted(special_tokens, key=len, reverse=True)
    if not specials:
        return [text]
    return re.split("(" + "|".join(map(re.escape, specials)) + ")", text)

def calculate_pairs(tokens: list[int]) -> dict[Pair, int]:
    pair_dict = defaultdict(int)
    for i in range(0, len(tokens) - 1):
        token_a, token_b = tokens[i], tokens[i+1]
        token_tuple = (token_a, token_b)
        pair_dict[token_tuple] += 1
    return pair_dict

def update_tokens(old_tokens: list[int], frequent_pair: Pair, newid: int) -> list[int]:
    new_tokens = list()
    
    i = 0
    while i < len(old_tokens):
        if i + 1 < len(old_tokens) and (old_tokens[i], old_tokens[i+1]) == frequent_pair:
            new_tokens.append(newid)
            i += 2
        else:
            new_tokens.append(old_tokens[i])
            i += 1
            
    return new_tokens

def train_bpe(corpus: str, vocab_size: int = 1000,
              special_tokens: list[str] | None = None) -> list[Merge]:
    """
    vocab_size 是最终词表总大小，包含特殊 token：
    merge 条数 = vocab_size - 256 - len(special_tokens)。
    特殊 token 不参与训练：语料先按它们切开并丢弃，因此任何 merge 都不会
    跨过、也不会包含特殊 token（包括语料里恰好出现的 "<|eos|>" 字面量）。

    增量式训练：
    - tokens/prev/nxt：双向链表，合并时 O(1) 删除右侧节点，不重建列表
    - counts：每个 pair 的真实计数
    - where：pair -> 出现位置（用左节点下标表示），合并时不用扫全量语料
    - heap：惰性删除的最大堆，只当"候选"，以 counts 为准

    预分词：先按 PRETOKEN_PATTERN 切片并统计 {词: 频次}，只对不同的词建链表，
    所有词拼在同一个数组里，词首 prev / 词尾 nxt 为 -1，pair 不会跨词。
    wt[i] 是位置 i 所在词的频次，pair 的计数按它加权。
    """
    special_tokens = special_tokens or []
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size 小于 256 + 特殊 token 数量")

    print("[INFO] Start Training")
    word_freq = Counter()
    for segment in split_special(corpus, special_tokens)[::2]:
        word_freq.update(PRETOKEN_PATTERN.findall(segment))

    tokens, prev, nxt, wt = [], [], [], []
    for word, freq in word_freq.items():
        ids = byte_encode(word)
        start, length = len(tokens), len(ids)
        tokens.extend(ids)
        wt.extend([freq] * length)
        prev.extend([-1, *range(start, start + length - 1)])
        nxt.extend([*range(start + 1, start + length), -1])

    counts = defaultdict(int)
    where = defaultdict(set)
    for i in range(len(tokens)):
        if nxt[i] != -1:
            pair = (tokens[i], tokens[nxt[i]])
            counts[pair] += wt[i]
            where[pair].add(i)

    heap = [(-cnt, pair) for pair, cnt in counts.items()]
    heapq.heapify(heap)

    touched = set()  # 本轮计数增加过的 pair，轮末统一入堆

    def dec(pair, pos):
        counts[pair] -= wt[pos]
        where[pair].discard(pos)

    def inc(pair, pos):
        counts[pair] += wt[pos]
        where[pair].add(pos)
        touched.add(pair)

    merges: list[Merge] = list()
    for newid in range(256, 256 + num_merges):
        pair = None
        while heap:
            neg_cnt, cand = heapq.heappop(heap)
            cnt = counts.get(cand, 0)
            if cnt <= 0 or -neg_cnt < cnt:
                # 已无效，或者堆里还有更新（更大）的条目
                continue
            if -neg_cnt > cnt:
                # 计数下降过，用真实计数重新入堆
                heapq.heappush(heap, (-cnt, cand))
                continue
            pair = cand
            break

        if pair is None:
            break

        merges.append((pair, newid))
        a, b = pair

        for i in sorted(where[pair]):
            j = nxt[i]
            # 重叠情况（如 aaa）下，位置可能已被前面的合并吞掉
            if tokens[i] != a or j == -1 or tokens[j] != b:
                continue
            p, nn = prev[i], nxt[j]

            dec(pair, i)
            if p != -1:
                dec((tokens[p], a), p)
            if nn != -1:
                dec((b, tokens[nn]), j)

            tokens[i] = newid
            tokens[j] = -1
            nxt[i] = nn
            if nn != -1:
                prev[nn] = i

            if p != -1:
                inc((tokens[p], newid), p)
            if nn != -1:
                inc((newid, tokens[nn]), i)

        del where[pair]
        del counts[pair]

        for touched_pair in touched:
            cnt = counts[touched_pair]
            if cnt > 0:
                heapq.heappush(heap, (-cnt, touched_pair))
        touched.clear()

    print("[INFO] Train Finished")

    return merges

def save_merges(merges: list[Merge], path: Path = MERGES_PATH) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for (token_a, token_b), newid in merges:
            file.write(f"{token_a} {token_b} {newid}\n")

def get_merges(path: Path = MERGES_PATH) -> list[Merge]:
    merges = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            token_a, token_b, newid = map(int, line.split())
            merges.append(((token_a, token_b), newid))

    assert merges
    return merges

def assign_special_ids(merges: list[Merge], special_tokens: list[str]) -> SpecialIds:
    """特殊 token 的 id 紧接在最后一条 merge 之后，不占用也不打乱 merge 的 id。"""
    if "" in special_tokens or len(set(special_tokens)) != len(special_tokens):
        raise ValueError("特殊 token 不能为空串，也不能重复")
    base = 256 + len(merges)
    return {token: base + k for k, token in enumerate(special_tokens)}

def build_vocab(merges: list[Merge], special_ids: SpecialIds | None = None) -> Vocab:
    """token id -> 原始 bytes。特殊 token 存它自己的 UTF-8 bytes，decode 时原样还原。"""
    vocab: Vocab = {i: bytes([i]) for i in range(256)}
    for (token_a, token_b), newid in merges:
        vocab[newid] = vocab[token_a] + vocab[token_b]
    for token, token_id in (special_ids or {}).items():
        assert token_id not in vocab, f"特殊 token id 冲突: {token_id}"
        vocab[token_id] = token.encode("utf-8")
    return vocab

def save_vocab(vocab: Vocab, special_ids: SpecialIds, path: Path = VOCAB_PATH) -> None:
    data = {
        "special_tokens": special_ids,
        "vocab": {str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()},
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)

def load_vocab(path: Path = VOCAB_PATH) -> tuple[Vocab, SpecialIds]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    vocab = {int(token_id): bytes.fromhex(hex_bytes) for token_id, hex_bytes in data["vocab"].items()}
    return vocab, data["special_tokens"]

def merge_piece(tokens: list[int], ranks: dict[Pair, int]) -> list[int]:
    # newid 越小，merge 学得越早；每轮只找当前序列里 rank 最小的相邻 pair 并合并
    inf = float("inf")
    while len(tokens) >= 2:
        best = min(zip(tokens, tokens[1:]), key=lambda pair: ranks.get(pair, inf))
        if best not in ranks:
            break
        tokens = update_tokens(tokens, best, ranks[best])
    return tokens

def bpe_encode(text: str, merges: list[Merge], special_ids: SpecialIds | None = None) -> list[int]:
    """
    special_ids 里的特殊 token 在文本中出现时直接映射成对应 id，不做 byte/BPE 处理。
    不传则所有文本（包括 "<|eos|>" 字面量）都按普通文本编码。
    """
    special_ids = special_ids or {}
    ranks = {pair: newid for pair, newid in merges}
    cache: dict[str, list[int]] = {}
    tokens: list[int] = []

    for idx, segment in enumerate(split_special(text, special_ids)):
        if idx % 2 == 1:
            tokens.append(special_ids[segment])
            continue
        for piece in PRETOKEN_PATTERN.findall(segment):
            if piece not in cache:
                cache[piece] = merge_piece(byte_encode(piece), ranks)
            tokens.extend(cache[piece])

    return tokens

def bpe_decode(tokens: list[int], vocab: Vocab) -> str:
    # 非法 UTF-8（比如只解码了半个字符的 token 序列）用 U+FFFD 代替，不抛异常
    return b"".join(vocab[token] for token in tokens).decode("utf-8", errors="replace")


def main():
    """
    从文件恢复：
    merges = get_merges()
    vocab, special_ids = load_vocab()
    """
    special_tokens = ["<|bos|>", "<|eos|>", "<|pad|>"]

    with open("/Users/WuShang/Documents/llm-from-zero/tokenizer/corpus.txt",
              "r",
              encoding="utf-8") as file:
        corpus = file.read()

    merges = train_bpe(corpus, 1000, special_tokens)
    special_ids = assign_special_ids(merges, special_tokens)
    vocab = build_vocab(merges, special_ids)

    save_merges(merges)
    save_vocab(vocab, special_ids)

    texts = ["中国", "Hello World\n!", "🤣🤣🤣", "<|bos|>Hello<|eos|><|pad|>"]

    for text in texts:
        assert bpe_decode(bpe_encode(text, merges, special_ids), vocab) == text


if __name__ == "__main__":
    main()
