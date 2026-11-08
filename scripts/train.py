import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import SequentialRecDataset
from models.dha_rec import DHARec
from models.baselines import SASRec, GRU4Rec
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="DHA-Rec Sequential Recommendation Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="MovieLens-1M", choices=["MovieLens-1M", "Amazon-Beauty", "Steam"])
    parser.add_argument("--model", type=str, default="dha_rec", choices=["dha_rec", "sasrec", "gru4rec"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()

def check_hardware_requirements():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n"
            "======================================================================\n"
            "HARDWARE ERROR: DHA-Rec requires an NVIDIA CUDA-enabled GPU to run.\n"
            "torch.cuda.is_available() returned False.\n"
            "CPU execution is disabled for high-throughput sequential attention.\n"
            "======================================================================\n"
        )
    print(f"[Hardware Check Passed] Using CUDA Device: {torch.cuda.get_device_name(0)}")

def main():
    args = parse_args()
    logger = setup_logger("DHA-Rec-Train")
    logger.info("Initializing DHA-Rec Sequential Recommendation Pipeline...")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model}")

    check_hardware_requirements()
    device = torch.device("cuda")

    data_path = os.path.join(args.data_dir, f"{args.dataset}.txt")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Please execute bash scripts/download_data.sh first.")

    train_dataset = SequentialRecDataset(data_path, max_len=args.max_len, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    num_items = train_dataset.num_items
    logger.info(f"Loaded {len(train_dataset)} user sequences across {num_items} items.")

    if args.model == "dha_rec":
        model = DHARec(num_items=num_items, hidden_dim=args.hidden_dim, max_len=args.max_len).to(device)
    elif args.model == "sasrec":
        model = SASRec(num_items=num_items, hidden_dim=args.hidden_dim, max_len=args.max_len).to(device)
    else:
        model = GRU4Rec(num_items=num_items, hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    bce_loss = nn.BCEWithLogitsLoss()

    logger.info(f"Starting training on {device} for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for batch_idx, (seq, pos, neg) in enumerate(train_loader):
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            
            # Forward pass: Will trigger tensor dimension mismatch on dha_rec!
            pos_logits, neg_logits = model(seq, pos, neg)
            
            loss = bce_loss(pos_logits, torch.ones_like(pos_logits)) + \
                   bce_loss(neg_logits, torch.zeros_like(neg_logits))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        elapsed = time.time() - start_time
        logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] Loss: {total_loss/len(train_loader):.4f} | Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
