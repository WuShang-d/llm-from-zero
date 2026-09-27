import torch
from model.transformer import TransformerLM
from tokenizer.bpe import (
    bpe_decode,
    bpe_encode,
    build_vocab,
)

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
seed = 42

def generate(
    model,
    prompt,
    max_tokens,
    vocab,
    merges,
    special_ids,
    temperature=1.0,
    top_k=50,
):
    assert max_tokens >= 0
    assert temperature >= 0
    assert 0 < top_k <= len(vocab)
    assert prompt

    token_ids = bpe_encode(prompt, merges, special_ids)

    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0) # [1, T]

    eos_id = special_ids["<|eos|>"]

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(tokens) # [B, T, V]
            next_logits = logits[:, -1] # [B, V]

            if temperature == 0:
                next_token = torch.argmax(
                    next_logits,
                    dim=-1,
                    keepdim=True,
                ) # [B, 1]

            else:
                next_logits = next_logits / temperature

                top_values, top_indices = torch.topk(
                    next_logits,
                    k=top_k,
                    dim=-1,
                )

                probabilities = torch.softmax(
                    top_values,
                    dim=-1,
                )

                sampled_index = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

                next_token = torch.gather(
                    top_indices,
                    dim=-1,
                    index=sampled_index,
                )
            tokens = torch.cat(
                [tokens, next_token],
                dim=1,
            )

            if next_token.item() == eos_id:
                break

    output_ids = tokens.squeeze(0).tolist()
    return bpe_decode(output_ids, vocab)

def main():
    # torch.manual_seed(seed)

    prompt = input("[PROMPT] User: ")
    while not prompt:
        print("Prompt 不能为空，请重新输入。")
        prompt = input("[PROMPT] User: ")

    checkpoint = torch.load(
        "checkpoints/best_model.pt",
        map_location=device,
    )

    model_config = checkpoint["model_config"]

    model = TransformerLM(
        vocab_size=model_config["vocab_size"],
        num_heads=model_config["num_heads"],
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer_config = checkpoint["tokenizer_config"]
    merges = tokenizer_config["merges"]
    special_ids = tokenizer_config["special_ids"]
    vocab = build_vocab(merges, special_ids)

    text = generate(
        model=model,
        prompt=prompt,
        max_tokens=220,
        vocab=vocab,
        merges=merges,
        special_ids=special_ids,
        temperature=1,
        top_k=50,
    )

    print(f"\n[Out] LM: {text}\n")


if __name__ == "__main__":
    main()
