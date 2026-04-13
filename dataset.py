import torch
import array
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset


class WikiText2Dataset(Dataset):
    def __init__(self, split, seq_len, dataset_variant="TinyStories", train_percent=100, max_tokens=None):
        """
        split: 'train', 'validation', or 'test'
        seq_len: length of each segment
        dataset_variant: dataset id or alias, e.g. TinyStories / roneneldan/TinyStories
        train_percent: percentage of train split to load when split='train'
        max_tokens: optional hard cap on number of tokens to keep
        """
        self.seq_len = seq_len
        self.dataset_variant = dataset_variant

        train_percent = int(max(1, min(100, train_percent)))
        split_expr = split
        if split == "train" and train_percent < 100:
            split_expr = f"train[:{train_percent}%]"

        # TinyStories provides train/validation; map test requests to validation.
        if split == "test":
            split_expr = "validation"

        # Load dataset
        source = "roneneldan/TinyStories" if self.dataset_variant in {"TinyStories", "roneneldan/TinyStories"} else self.dataset_variant
        raw = load_dataset(source, split=split_expr, streaming=True)

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=10_000_000)
        self.tokenizer.model_max_length = 10_000_000

        # Encode incrementally so we never materialize the full dataset text in RAM.
        token_buffer = array.array("I")
        token_limit = None if max_tokens is None else int(max(2, max_tokens))

        for example in raw:
            text = example.get("text", "")
            if not text or not text.strip():
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue

            if token_limit is None:
                token_buffer.extend(ids)
                continue

            remaining = token_limit - len(token_buffer)
            if remaining <= 0:
                break

            if len(ids) <= remaining:
                token_buffer.extend(ids)
            else:
                token_buffer.extend(ids[:remaining])
                break

        # Keep storage compact; cast to long on retrieval for the embedding layer.
        self.data = torch.tensor(token_buffer, dtype=torch.int32)

    def __len__(self):
        # number of segments
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.data[start:end].to(dtype=torch.long)

        x = chunk[:-1]   # input
        y = chunk[1:]    # target (shifted)

        return x, y 