import os
import torch
from torch.utils.data import Dataset
import random

class SequentialRecDataset(Dataset):
    """
    Dataset for sequential recommendation with negative sampling.
    """
    def __init__(self, data_path, max_len=50, split="train", num_neg=1):
        self.max_len = max_len
        self.split = split
        self.num_neg = num_neg

        if not os.path.exists(data_path):
            try:
                from .download_data import generate_benchmark_data
                dataset_name = os.path.splitext(os.path.basename(data_path))[0]
                os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
                generate_benchmark_data(dataset_name, data_path)
            except Exception:
                pass

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file '{data_path}' not found! Run scripts/download_data.sh first.")

        self.user_seqs = {}
        self.num_items = 0

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [int(x) for x in line.strip().split()]
                if len(parts) < 3:
                    continue
                user_id = parts[0]
                seq = parts[1:]
                self.user_seqs[user_id] = seq
                self.num_items = max(self.num_items, max(seq))

        self.users = list(self.user_seqs.keys())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        seq = self.user_seqs[user_id]

        if self.split == "train":
            in_seq = seq[:-2]
            target_pos = seq[-2]
        elif self.split == "val":
            in_seq = seq[:-2]
            target_pos = seq[-2]
        else: # test
            in_seq = seq[:-1]
            target_pos = seq[-1]

        # Truncate / pad to max_len
        pad_len = self.max_len - len(in_seq)
        if pad_len > 0:
            padded_seq = [0] * pad_len + in_seq
        else:
            padded_seq = in_seq[-self.max_len:]

        # Random negative item
        neg_item = random.randint(1, self.num_items)
        while neg_item in seq:
            neg_item = random.randint(1, self.num_items)

        return (
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )
