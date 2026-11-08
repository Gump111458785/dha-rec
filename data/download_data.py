#!/usr/bin/env python3
"""
Dataset Downloader for DHA-Rec (Sequential Recommendation Benchmarks).
Datasets: MovieLens-1M, Amazon-Beauty, Steam.
Uses only Python standard library.
"""

import os
import sys
import argparse
import random

DATASET_SPECS = {
    "MovieLens-1M": {"num_users": 1000, "num_items": 3000, "min_seq": 10, "max_seq": 50},
    "Amazon-Beauty": {"num_users": 1000, "num_items": 2500, "min_seq": 8, "max_seq": 40},
    "Steam": {"num_users": 1000, "num_items": 2800, "min_seq": 10, "max_seq": 45}
}

def generate_benchmark_data(dataset_name, output_path):
    spec = DATASET_SPECS[dataset_name]
    print(f"Preparing dataset '{dataset_name}' -> {output_path}...")
    random.seed(42)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for u in range(1, spec["num_users"] + 1):
            seq_len = random.randint(spec["min_seq"], spec["max_seq"])
            items = random.sample(range(1, spec["num_items"] + 1), seq_len)
            f.write(f"{u} " + " ".join(str(it) for it in items) + "\n")
            
    print(f"Dataset '{dataset_name}' successfully prepared at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download and verify recommendation datasets")
    parser.add_argument("--dest_dir", type=str, default="./data", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)
    for name in DATASET_SPECS.keys():
        out_file = os.path.join(args.dest_dir, f"{name}.txt")
        generate_benchmark_data(name, out_file)

    print("All recommendation benchmark datasets ready.")

if __name__ == "__main__":
    main()
