import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ChordBeatEncoder(nn.Module):
    """
    Input per timestep:
        [12 chroma | 12 bass | 4 beat]
        = 28 features

    Input shape:
        (B, T, 28)

    Output:
        z : (B, d_model)  # projected embedding
        h : (B, d_model)  # pooled transformer representation
    """

    def __init__(
        self,
        input_dim=28,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()

        # Project 28-dimensional symbolic features into transformer space
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Attention pooling
        self.pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1),
        )

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, 28)

        Returns:
            z: (B, d_model)
            h: (B, d_model)
        """

        # (B, T, 28) -> (B, T, D)
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        x = self.transformer(x)

        # Attention pooling across time
        weights = self.pool(x)          # (B, T, 1)
        h = (x * weights).sum(dim=1)    # (B, D)

        # Final embedding
        z = self.output_proj(h)

        return z, h