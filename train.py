import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from config import Config
from model import TransformerXL
from dataset import WikiText2Dataset


def train():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    grad_accum_steps = max(1, getattr(config, "grad_accum_steps", 1))

    # Initialize wandb
    use_wandb = config.use_wandb and wandb is not None
    if use_wandb:
        wandb.init(project="Transformer-XL", config={
            "dataset_variant": config.dataset_variant,
            "train_percent": config.train_percent,
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "batch_size": config.batch_size,
            "grad_accum_steps": grad_accum_steps,
            "seq_len": config.seq_len,
            "lr": config.lr,
            "epochs": config.epochs
        })

    # Dataset
    train_dataset = WikiText2Dataset(
        "train",
        config.seq_len,
        dataset_variant=config.dataset_variant,
        train_percent=config.train_percent,
        max_tokens=getattr(config, "max_train_tokens", None),
    )
    val_dataset = WikiText2Dataset(
        "validation",
        config.seq_len,
        dataset_variant=config.dataset_variant,
        train_percent=100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # IMPORTANT for Transformer-XL
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True
    )

    # Model
    model = TransformerXL(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        mem_len=config.mem_len,
        dropout=config.dropout
    ).to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.lr)

    # Mixed precision
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=config.use_amp)

    # Scheduler & Logging
    total_optimizer_steps = config.epochs * ((len(train_loader) + grad_accum_steps - 1) // grad_accum_steps)
    warmup_steps = min(getattr(config, "warmup_steps", 1000), max(total_optimizer_steps - 1, 0))
    min_lr_ratio = getattr(config, "min_lr_ratio", 0.1)

    def lr_lambda(step):
        if total_optimizer_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_val_loss = float('inf')
    start_epoch = 0
    global_step = 0

    if hasattr(config, 'resume_checkpoint') and config.resume_checkpoint:
        print(f"Loading checkpoint from {config.resume_checkpoint}...")
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("val_loss", float('inf'))
        global_step = checkpoint.get("global_step", 0)
        print(f"Resuming from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    
    # Optional: Log to file
    log_mode = "a" if start_epoch > 0 else "w"
    log_file = open("training_log.txt", log_mode)
    if start_epoch == 0:
        log_file.write("Epoch\tTrain_Loss\tTrain_PPL\tVal_Loss\tVal_PPL\n")

    for epoch in range(start_epoch, config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        model.train()

        mems = None
        mem_batch_size = None

        total_loss = 0
        optimizer.zero_grad(set_to_none=True)

        loop = tqdm(train_loader)

        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(device)
            y = y.to(device)

            if mems is None or mem_batch_size != x.size(0):
                mems = model.init_mems(x.size(0))
                mem_batch_size = x.size(0)

            with torch.amp.autocast(amp_device, enabled=config.use_amp):
                logits, mems = model(x, mems)

                loss = criterion(
                    logits.view(-1, config.vocab_size),
                    y.view(-1)
                )

            if not torch.isfinite(loss):
                print("Encountered non-finite loss; skipping batch.")
                mems = model.init_mems(x.size(0))
                mem_batch_size = x.size(0)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            total_loss += loss.item()
            if use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                })

            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        ppl = torch.exp(torch.tensor(avg_loss))

        print(f"Train Loss: {avg_loss:.4f}, Perplexity: {ppl:.2f}")

        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        val_loss = 0

        mems = None
        mem_batch_size = None

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                if mems is None or mem_batch_size != x.size(0):
                    mems = model.init_mems(x.size(0))
                    mem_batch_size = x.size(0)

                logits, mems = model(x, mems)

                loss = criterion(
                    logits.view(-1, config.vocab_size),
                    y.view(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_ppl = torch.exp(torch.tensor(avg_val_loss))

        print(f"Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_ppl:.2f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_ppl": ppl.item(),
                "val_loss": avg_val_loss,
                "val_ppl": val_ppl.item()
            })

        log_file.write(f"{epoch+1}\t{avg_loss:.4f}\t{ppl:.2f}\t{avg_val_loss:.4f}\t{val_ppl:.2f}\n")
        log_file.flush()

        # -----------------------
        # Save checkpoint
        # -----------------------
        state_dict = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "val_loss": avg_val_loss
        }
        torch.save(state_dict, f"checkpoint_epoch_{epoch+1}.pt")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(state_dict, "best_checkpoint.pt")
            print(f"--> Saved new best checkpoint with val loss {best_val_loss:.4f}")

    log_file.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()