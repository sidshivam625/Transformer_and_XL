import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from config import Config
from model import TransformerXL

class TinyDataset(Dataset):
    def __init__(self, seq_len, repeats=1000):
        self.seq_len = seq_len

        pattern = [1, 2, 3, 2]  # tiny deterministic pattern to force quick overfit
        tokens = pattern * repeats
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def test_overfit():
    print("Starting Tiny Overfit Test...")
    config = Config()
    
    # We might want to use a smaller model or fewer epochs for a quick test
    # But let's stick to the default config structure or override few things
    config.batch_size = 4
    config.seq_len = 16
    config.mem_len = 16
    config.epochs = 50
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    train_dataset = TinyDataset(config.seq_len)
    config.vocab_size = int(train_dataset.data.max().item()) + 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True
    )

    model = TransformerXL(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        mem_len=config.mem_len,
        dropout=config.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-4)

    for epoch in range(config.epochs):
        model.train()
        mems = None
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if mems is None:
                mems = model.init_mems(x.size(0))
            optimizer.zero_grad()
            
            logits, mems = model(x, mems)
            
            loss = criterion(
                logits.view(-1, config.vocab_size),
                y.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_loss:.4f}")
            
        if avg_loss < 0.05:
            print(f"Overfitting successful at epoch {epoch+1}! Loss is {avg_loss:.4f}")
            break

if __name__ == "__main__":
    test_overfit()
