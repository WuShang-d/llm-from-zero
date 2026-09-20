"""
    A stupid tokenizer
"""

def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))

def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")

texts = ["中国", "Hello World\n!", "🤣🤣🤣"]

for text in texts:
    print(text, encode(text))
    print("\n")
    assert decode(encode(text)) == text
