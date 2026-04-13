import math
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import WikiText2Dataset
from model import TransformerXL


def load_model(device):
    config = Config()
    model = TransformerXL(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        mem_len=config.mem_len,
        dropout=config.dropout,
    ).to(device)

    checkpoint = torch.load(config.resume_checkpoint or config.checkpoint_path, map_location=device)
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(model_state)
    model.eval()
    return model, config


def evaluate(split="test"):
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model, config = load_model(device)

    dataset = WikiText2Dataset(split, config.seq_len)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0

    mems = model.init_mems(config.batch_size)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, mems = model(x, mems)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    ppl = math.exp(avg_loss)

    metrics = {
        "split": split,
        "loss": avg_loss,
        "ppl": ppl,
    }

    metrics_path = Path("evaluation_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"{split.title()} Loss: {avg_loss:.4f}")
    print(f"{split.title()} Perplexity: {ppl:.2f}")
    print(f"Saved evaluation metrics to {metrics_path}")
    return avg_loss, ppl


if __name__ == "__main__":
    evaluate("test")