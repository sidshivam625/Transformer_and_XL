import torch

class Config:
    # Model
    vocab_size = 50257
    d_model = 320
    n_heads = 5
    n_layers = 7
    d_ff = 1280
    mem_len = 144
    dropout = 0.1

    # Training
    batch_size = 7
    seq_len = 128
    lr = 2e-4
    epochs = 5
    resume_checkpoint = None # e.g., "checkpoint_epoch_2.pt" or "best_checkpoint.pt"
    use_amp = True # mixed precision training
    use_wandb = True
    checkpoint_path = "best_checkpoint.pt"
    

    # Optimization
    grad_clip = 0.25

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    debug = False # for debug mode. If True, will use smaller dataset and fewer epochs for quick testing.