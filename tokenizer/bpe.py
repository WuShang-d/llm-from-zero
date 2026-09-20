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

from collections import defaultdict

def byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))

def byte_decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")

def calculate_pairs(tokens: list[int]):
    pair_dict = defaultdict(int)
    for i in range(0, len(tokens) - 1):
        token_a, token_b = tokens[i], tokens[i+1]
        token_tuple = (token_a, token_b)
        pair_dict[token_tuple] += 1
    return pair_dict

def update_tokens(old_tokens: list[int], frequent_pair: tuple, newid: int):
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

def train_bpe(corpus, vocab_size=1000):
    print("[INFO] Start Training")
    tokens = byte_encode(corpus)
    merges = list()
    for i in range(256, vocab_size):
        pair_dict = calculate_pairs(tokens)
        newid, pair, curr = i, None, 0
        
        for pairs, cnt in pair_dict.items():
            if cnt > curr:
                pair = pairs
                curr = cnt
        
        if pair is None:
            break
        
        merges.append((pair, newid))
        
        tokens = update_tokens(tokens, pair, newid)
    
    print("[INFO] Train Finished")
    
    return merges

with open("/Users/WuShang/Documents/llm-from-zero/tokenizer/corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read()
    
merges = train_bpe(corpus)

with open("merges.txt", "w", encoding="utf-8") as file:
    for pair, newid in merges:
        token_a, token_b = pair
        file.write(f"{token_a} {token_b} {newid}\n")
        
"""
    从文件恢复 mreges：
    merges = []

    with open("merges.txt", "r", encoding="utf-8") as file:
        for line in file:
            token_a, token_b, newid = map(int, line.split())
            merges.append(((token_a, token_b), newid))
"""