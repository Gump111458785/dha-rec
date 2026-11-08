import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

class DHARec(nn.Module):
    """
    DHA-Rec: Dual-Horizon Attentive Recommendation Network.
    Decouples short-term dynamic intent from long-term preference horizons.
    
    [BENCHMARK ANOMALY INJECTION]
    NOTE: This repository contains an intentional tensor dimension mismatch bug in the
    cross-horizon fusion module to test automated reproduction auditing engines.
    """
    def __init__(self, num_items=3000, hidden_dim=64, max_len=50, num_heads=2):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)

        # Short-term attention block
        self.short_term_attn = MultiHeadAttentionBlock(hidden_dim, num_heads)
        # Long-term attention block
        self.long_term_attn = MultiHeadAttentionBlock(hidden_dim, num_heads)

        # Intentional Bug Injection: Incompatible inner projection dimension (128 vs 64)
        self.fusion_matrix = nn.Parameter(torch.randn(128, hidden_dim) * 0.02)
        self.classifier = nn.Linear(hidden_dim, num_items + 2)

    def forward(self, seq, pos_items=None, neg_items=None):
        # seq: [B, L]
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).repeat(B, 1)

        x = self.item_embedding(seq) + self.pos_embedding(positions)
        padding_mask = (seq == 0)

        # Short-term slice (last 10 items)
        short_x = x[:, -10:, :]
        short_mask = padding_mask[:, -10:]
        short_repr = self.short_term_attn(short_x, short_mask)[:, -1, :] # [B, 64]

        # Long-term global sequence
        long_repr = self.long_term_attn(x, padding_mask)[:, -1, :] # [B, 64]

        # [FATAL REPOSITORY BUG]
        # Attempting matmul between short_repr [B, 64] and fusion_matrix [128, 64]
        # Raises RuntimeError: mat1 and mat2 shapes cannot be multiplied!
        fusion_out = torch.matmul(short_repr, self.fusion_matrix)

        final_repr = fusion_out + long_repr

        if pos_items is not None and neg_items is not None:
            pos_emb = self.item_embedding(pos_items) # [B, 64]
            neg_emb = self.item_embedding(neg_items) # [B, 64]
            pos_logits = (final_repr * pos_emb).sum(dim=-1)
            neg_logits = (final_repr * neg_emb).sum(dim=-1)
            return pos_logits, neg_logits

        return final_repr
