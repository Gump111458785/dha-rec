import time
import torch
from models.baselines import SASRec, GRU4Rec

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU device is required.")
    device = torch.device("cuda")

    B, L = 32, 50
    seq = torch.randint(1, 2000, (B, L), device=device)

    print("=" * 60)
    print("DHA-Rec Efficiency Benchmark (Latency & Parameters)")
    print("=" * 60)
    models = {
        "SASRec": SASRec(num_items=3000, hidden_dim=64, max_len=50).to(device),
        "GRU4Rec": GRU4Rec(num_items=3000, hidden_dim=64).to(device)
    }

    for name, m in models.items():
        m.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = m(seq)
        lat = (time.time() - start) * 1000 / 50
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name:<12} | Params: {params/1e3:.1f} K | Latency: {lat:.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    main()
