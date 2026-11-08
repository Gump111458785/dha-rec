import os
import argparse
import json
import torch
from torch.utils.data import DataLoader

from data.dataset import SequentialRecDataset
from models.dha_rec import DHARec
from utils.logger import setup_logger
from utils.metrics import compute_topk_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate DHA-Rec")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="MovieLens-1M")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best.pt")
    args = parser.parse_args()

    logger = setup_logger("DHA-Rec-Eval")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU device is required for evaluation.")

    print("Evaluating DHA-Rec sequential performance...")

if __name__ == "__main__":
    main()
