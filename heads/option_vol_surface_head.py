import torch
import torch.nn as nn

class VolSurfaceHead(nn.Module):
    """
    Maps Transformer encoder outputs to implied volatility surface predictions.
    Typically a small MLP.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)

        Returns:
            (batch_size, seq_len)
        """
        out = self.net(x)  # (batch_size, seq_len, 1)
        out = out.squeeze(-1)  # Remove last dimension
        return out