import numpy as np

def compute_topk_metrics(ranks, k_list=[5, 10, 20]):
    """
    Computes HitRate@K, NDCG@K, MRR@K.
    """
    results = {}
    for k in k_list:
        hit = np.mean([1.0 if r <= k else 0.0 for r in ranks])
        ndcg = np.mean([1.0 / np.log2(r + 1) if r <= k else 0.0 for r in ranks])
        mrr = np.mean([1.0 / r if r <= k else 0.0 for r in ranks])
        results[f"HR@{k}"] = round(float(hit), 4)
        results[f"NDCG@{k}"] = round(float(ndcg), 4)
        results[f"MRR@{k}"] = round(float(mrr), 4)
    return results
