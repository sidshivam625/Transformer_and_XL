import torch

class Config:
    # Data
    dataset_variant = "TinyStories"  # resolved to roneneldan/TinyStories in dataset loader
    train_percent = 30  # used only for train split when < 100
    max_train_tokens = 20_000_000  # roughly 20M training tokens
    max_eval_tokens = 2_000_000  # keep eval memory bounded in notebooks

    # Model
    vocab_size = 50257
    d_model = 512
    n_heads = 8
    n_layers = 8
    d_ff = 2048
    mem_len = 192
    dropout = 0.1

    # Training
    batch_size = 8
    seq_len = 192
    lr = 2.5e-4
    epochs = 12
    grad_accum_steps = 2
    warmup_steps = 1000
    min_lr_ratio = 0.1
    resume_checkpoint = None # e.g., "checkpoint_epoch_2.pt" or "best_checkpoint.pt"
    use_amp = True # mixed precision training
    use_wandb = True
    checkpoint_path = "best_checkpoint.pt"
    

    # Optimization
    grad_clip = 0.25

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    debug = False # for debug mode. If True, will use smaller dataset and fewer epochs for quick testing.