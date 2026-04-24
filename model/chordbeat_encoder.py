import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)
    

class ChordBeatEncoder(nn.Module):
    def __init__(self, input_dim=13, d_model=128, nhead=4, num_layers=3):
        super().__init__()

        # project input → model dim
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # pooling (same idea as your audio side)
        self.pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (B, T, 13)
        """

        x = self.input_proj(x)      # (B, T, D)
        x = self.pos_enc(x)

        x = self.transformer(x)     # (B, T, D)

        # attention pooling
        weights = self.pool(x)      # (B, T, 1)
        h = (x * weights).sum(dim=1)

        z = self.output_proj(h)

        return z, h