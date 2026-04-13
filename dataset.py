import torch
import array
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset, load_dataset_builder


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

        # TinyStories provides train/validation; map test requests to validation.
        if split == "test":
            split_expr = "validation"

        # Load dataset
        source = "roneneldan/TinyStories" if self.dataset_variant in {"TinyStories", "roneneldan/TinyStories"} else self.dataset_variant
        raw = load_dataset(source, split=split_expr, streaming=True)

        max_examples = None
        if split_expr == "train" and train_percent < 100:
            # Streaming mode does not support split slicing like train[:30%].
            # Estimate the number of examples from split metadata and stop early.
            try:
                builder = load_dataset_builder(source)
                split_info = builder.info.splits.get("train") if builder.info and builder.info.splits else None
                if split_info is not None and getattr(split_info, "num_examples", None):
                    max_examples = max(1, int(split_info.num_examples * (train_percent / 100.0)))
            except Exception:
                max_examples = None

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=10_000_000)
        self.tokenizer.model_max_length = 10_000_000

        # Encode incrementally so we never materialize the full dataset text in RAM.
        token_buffer = array.array("I")
        token_limit = None if max_tokens is None else int(max(2, max_tokens))

        examples_seen = 0
        for example in raw:
            if max_examples is not None and examples_seen >= max_examples:
                break

            examples_seen += 1

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