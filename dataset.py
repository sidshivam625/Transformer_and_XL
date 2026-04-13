import torch
import warnings
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset


class WikiText2Dataset(Dataset):
    def __init__(self, split, seq_len):
        """
        split: 'train', 'validation', or 'test'
        seq_len: length of each segment
        """
        self.seq_len = seq_len

        # Load dataset
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")

        # Remove empty lines and join into one long string
        text = "\n".join(
            [t for t in raw[split]["text"] if t.strip() != ""]
        )

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", model_max_length=10_000_000)
        self.tokenizer.model_max_length = 10_000_000

        # Encode entire corpus
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Token indices sequence length is longer than the specified maximum sequence length.*",
            )
            tokens = self.tokenizer.encode(text)

        # Convert to tensor
        self.data = torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        # number of segments
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = self.data[start:end]

        x = chunk[:-1]   # input
        y = chunk[1:]    # target (shifted)

        return x, y 