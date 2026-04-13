import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

from config import Config
from model import TransformerXL


def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    """
    🔥 IMPROVEMENT 1: Top-k + nucleus (top-p) filtering
    - Top-k: keep only top k tokens
    - Top-p: keep smallest set of tokens with cumulative prob >= p
    """

    # Top-k
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -1e9), logits)

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )

        logits = logits.masked_fill(indices_to_remove, -1e9)

    return logits


def apply_repetition_penalty(logits, generated, penalty=1.2):
    """
    🔥 IMPROVEMENT 2: Repetition penalty
    - discourages repeating the same tokens
    """

    for token in set(generated.tolist()):
        logits[:, token] /= penalty

    return logits


def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2
):
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = TransformerXL(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        mem_len=config.mem_len,
        dropout=0.0
    ).to(device)

    checkpoint_path = config.resume_checkpoint or config.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(model_state)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    mems = None
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # forward with memory
            logits, mems = model(generated[:, -1:], mems)

            logits = logits[:, -1, :]

            # 🔥 IMPROVEMENT 3: temperature scaling
            logits = logits / temperature

            # 🔥 IMPROVEMENT 4: repetition penalty
            logits = apply_repetition_penalty(logits, generated[0], repetition_penalty)

            # 🔥 IMPROVEMENT 5: top-k + top-p filtering
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

    output = tokenizer.decode(generated[0].tolist())

    return output


if __name__ == "__main__":
    prompt = "The meaning of life is"

    text = generate(
        prompt,
        max_new_tokens=100,
        temperature=0.8,      # 🔥 controls randomness
        top_k=50,             # 🔥 limits candidate tokens
        top_p=0.9,            # 🔥 nucleus sampling
        repetition_penalty=1.2  # 🔥 avoids loops
    )

    print(text) 