import torch
import torch.nn as nn


class AttentionWeightedPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attn(x)
        weighted = x * weights
        pooled = weighted.sum(dim=1) / (weights.sum(dim=1) + 1e-8)
        return pooled
