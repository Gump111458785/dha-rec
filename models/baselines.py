import torch
import torch.nn as nn

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (ICDM 2018)
    """
    def __init__(self, num_items=3000, hidden_dim=64, max_len=50, num_heads=2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq, pos_items=None, neg_items=None):
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).repeat(B, 1)
        x = self.item_embedding(seq) + self.pos_embedding(pos)
        attn_out, _ = self.attn(x, x, x)
        repr_out = self.norm(attn_out)[:, -1, :]

        if pos_items is not None and neg_items is not None:
            p_emb = self.item_embedding(pos_items)
            n_emb = self.item_embedding(neg_items)
            return (repr_out * p_emb).sum(-1), (repr_out * n_emb).sum(-1)
        return repr_out

class GRU4Rec(nn.Module):
    """
    Session-based Recommendation with Recurrent Neural Networks (ICLR 2016)
    """
    def __init__(self, num_items=3000, hidden_dim=64):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 2, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, seq, pos_items=None, neg_items=None):
        x = self.item_embedding(seq)
        out, _ = self.gru(x)
        repr_out = out[:, -1, :]

        if pos_items is not None and neg_items is not None:
            p_emb = self.item_embedding(pos_items)
            n_emb = self.item_embedding(neg_items)
            return (repr_out * p_emb).sum(-1), (repr_out * n_emb).sum(-1)
        return repr_out
