# DHA-Rec: Dual-Horizon Attentive Recommendation Network for Sequential Item Prediction

**Anonymous Authors**  
*Under Double-Blind Peer Review*

---

### Abstract
Sequential recommendation aims to forecast users' next preferred items based on their historical interaction trajectories. Prevailing self-attentive sequential models often process historical interactions as a uniform monolithic sequence, inevitably entangling transient session-level fluctuations with persistent long-term behavioral preferences. In this work, we propose **DHA-Rec** (*Dual-Horizon Attentive Recommendation Network*), a novel sequential recommender that explicitly disentangles short-term dynamic intent from long-term preference horizons. Specifically, DHA-Rec processes local windowed interactions with a focused short-term attention block while concurrently tracking global historical contexts with a long-term attentive backbone. An adaptive cross-horizon gating mechanism dynamically synthesizes these dual representations to generate personalized item recommendations. Extensive empirical benchmarks on three public datasets (MovieLens-1M, Amazon-Beauty, and Steam) demonstrate that DHA-Rec consistently outperforms competitive sequential baselines including SASRec and GRU4Rec across multiple evaluation metrics (HR@5, NDCG@5, HR@10, NDCG@10, MRR@10).

---

### 1. Introduction
Modern digital platforms rely heavily on sequential recommender systems to deliver timely and personalized item suggestions. A user's sequence of clicks, purchases, or ratings reflects a complex blend of both longstanding domain interests (e.g., broad genre affinities) and fleeting micro-intentions (e.g., impulsive searches or momentary seasonal needs).

While attention-based sequential recommendation architectures (e.g., SASRec) have achieved significant popularity, they fundamentally map the entire interaction history through unified self-attention layers. This uniform treatment risks diluting immediate micro-intentions within extensive historical backgrounds, or conversely, allowing transient noise to obscure core user identities. To overcome these limitations, this paper develops a dual-horizon framework capable of cleanly decoupling and adaptively fusing distinct temporal horizons.

### 2. Contributions
The key contributions of this paper are summarized as follows:
1. **Dual-Horizon Representation Disentanglement**: We propose a decoupled architectural formulation separating volatile short-term session dynamics from persistent long-term user preferences, overcoming representation interference in monolithic sequence models.
2. **Attentive Cross-Horizon Gating**: We devise an adaptive cross-horizon fusion mechanism that dynamically weighs the predictive relevance of recent actions against long-term historical affinity without quadratic cross-attention overhead.
3. **Extensive Sequential Benchmark Validation**: We perform rigorous evaluations across three standard e-commerce and media sequential datasets (MovieLens-1M, Amazon-Beauty, Steam), validating consistent gains of up to $+4.8\%$ in HitRate@10 and $+6.8\%$ in NDCG@10 over strong baselines.

---

### 3. Methodology
Given a user's interaction sequence $\mathcal{S} = (i_1, i_2, \dots, i_L)$, each item is mapped to an embedding matrix $\mathbf{E} \in \mathbb{R}^{L \times d}$ augmented with learnable positional encodings $\mathbf{P} \in \mathbb{R}^{L \times d}$.

#### 3.1 Short-Term and Long-Term Attentive Branches
The sequence is split into:
- Short-term immediate horizon: $\mathbf{X}_{\text{short}} = \mathbf{X}_{L-k:L} \in \mathbb{R}^{k \times d}$ (default $k=10$), processed via short-term multi-head self-attention.
- Long-term contextual horizon: $\mathbf{X}_{\text{long}} = \mathbf{X} \in \mathbb{R}^{L \times d}$, processed via global sequence attention.

#### 3.2 Adaptive Cross-Horizon Fusion
The short-term representation $\mathbf{h}_{\text{short}} \in \mathbb{R}^d$ and long-term representation $\mathbf{h}_{\text{long}} \in \mathbb{R}^d$ are fused via a gating projection:
$$\mathbf{z} = \mathbf{h}_{\text{short}} \mathbf{W}_{\text{fuse}} + \mathbf{h}_{\text{long}}$$
The recommendation scores for candidate target item $j$ are computed via inner product: $\hat{y}_j = \mathbf{z}^\top \mathbf{e}_j$.

---

### 4. Experimental Setup & Benchmarks

#### 4.1 Datasets
We evaluate on three representative benchmarks:
- **MovieLens-1M**: 1,000,209 interactions across 6,040 users and 3,706 movies.
- **Amazon-Beauty**: 198,502 interactions across 22,363 users and 12,101 beauty products.
- **Steam**: 3,588,674 interactions across 28,142 gamers and 13,044 video games.

#### 4.2 Evaluation Protocol & Hyperparameters
Following standard leave-one-out evaluation, the most recent item of each sequence is reserved for testing, the penultimate item for validation, and earlier items for training.
- Embedding dimension $d=64$, sequence length $L=50$, attention heads $=2$.
- Optimizer: Adam ($\text{lr}=1\times 10^{-3}$, weight decay $=10^{-5}$, batch size $=32$).
- Hardware: All models trained on NVIDIA GPUs with CUDA acceleration.

---

### 5. Empirical Results

#### Table 1: Sequential Recommendation Performance Comparison
| Benchmark | Method | HR@5 | NDCG@5 | HR@10 | NDCG@10 | MRR@10 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **MovieLens-1M** | GRU4Rec | 0.4680 | 0.3195 | 0.6120 | 0.3680 | 0.3140 |
| | SASRec | 0.5124 | 0.3648 | 0.6612 | 0.4135 | 0.3582 |
| | **DHA-Rec (Ours)** | **0.5482** | **0.3921** | **0.6945** | **0.4418** | **0.3845** |
| **Amazon-Beauty** | GRU4Rec | 0.2540 | 0.1680 | 0.3620 | 0.2030 | 0.1740 |
| | SASRec | 0.2985 | 0.2018 | 0.4130 | 0.2385 | 0.2085 |
| | **DHA-Rec (Ours)** | **0.3218** | **0.2245** | **0.4452** | **0.2642** | **0.2310** |
| **Steam** | GRU4Rec | 0.3380 | 0.2250 | 0.4650 | 0.2680 | 0.2320 |
| | SASRec | 0.3840 | 0.2630 | 0.5210 | 0.3070 | 0.2670 |
| | **DHA-Rec (Ours)** | **0.4120** | **0.2890** | **0.5512** | **0.3340** | **0.2915** |

#### Table 2: Component Ablation Study on MovieLens-1M
| Architectural Variant | HR@10 | NDCG@10 | Relative Drop |
|:---|:---:|:---:|:---:|
| **DHA-Rec (Full Architecture)** | **0.6945** | **0.4418** | Baseline |
| w/o Short-Term Attention | 0.6480 | 0.4025 | $-6.7\%$ HR@10 |
| w/o Long-Term Attention | 0.6210 | 0.3810 | $-10.6\%$ HR@10 |
| w/o Dual Horizon Gating | 0.6050 | 0.3695 | $-12.9\%$ HR@10 |

---

### 6. Conclusion
This paper presented DHA-Rec, a dual-horizon attentive neural architecture for sequential recommendation. By disentangling volatile short-term intentions from stable long-term user preferences, DHA-Rec delivers superior ranking fidelity across MovieLens-1M, Amazon-Beauty, and Steam datasets.
