import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWeightedPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()

        # equivalent to conv blocks in paper → here MLP over time
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, T, C)
        """
        # compute attention weights
        weights = self.attn(x)  # (B, T, 1)

        # apply weights
        weighted = x * weights  # (B, T, C)

        # weighted average pooling
        pooled = weighted.sum(dim=1) / (weights.sum(dim=1) + 1e-8)

        return pooled  # (B, C)