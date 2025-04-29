import torch
import torch.nn as nn
from embeddings.positional_encoding import PositionalEncoding

class MaskedGeneralInputProcessor(nn.Module):
    """
    Masked General Input Processor for variable asset IV surfaces.

    This module processes batched tabular inputs where the number of assets may vary across samples.
    It dynamically identifies real (non-padded) input dimensions for each sample based on the provided asset_dims list.
    Only the real strike and maturity features are embedded (ignoring padded dimensions), and the resulting embeddings
    are averaged across available features to form a single embedding per token.

    Args:
        input_dim (int): Total input dimension (max number of assets + maturity).
        embed_dim (int): Output embedding dimension for the Transformer.
        use_mlp (bool, optional): If True, use a shallow MLP (2-layer) for per-feature projection instead of a single Linear layer. Default: False.
        use_positional (bool, optional): If True, add sinusoidal positional encodings to the embeddings. Default: True.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim), potentially containing padded zeros.
        asset_dims (list[int]): List of length batch_size, where each entry indicates the number of real assets for that sample
                                (i.e., input_dim = num_assets + 1 maturity).

    Returns:
        out (torch.Tensor): Output embedded tensor of shape (batch_size, seq_len, embed_dim).

    Notes:
        - Real strike/maturity features are embedded individually.
        - The embeddings are averaged across real features before feeding into Transformer.
        - Padded dimensions are ignored completely before embedding.
        - This approach reduces noise from padding but requires slightly more computational overhead.
    """
    def __init__(self, input_dim, embed_dim, use_mlp=False, use_positional=True):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_positional = use_positional

        if use_mlp:
            self.raw_projector = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.raw_projector = nn.Linear(1, embed_dim)

        if self.use_positional:
            self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x: torch.Tensor, asset_dims: list[int]) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            asset_dims: list[int], number of real assets per sample

        Returns:
            (batch_size, seq_len, embed_dim)
        """
        assert x.ndim == 3, f"Expected input shape (B, T, D), got {x.shape}"
        assert len(asset_dims) == x.size(0), f"asset_dims length {len(asset_dims)} doesn't match batch size {x.size(0)}"

        batch_size, seq_len, _ = x.shape
        out = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)

        for i in range(batch_size):
            num_assets = asset_dims[i]
            real_features = x[i, :, :num_assets + 1]  # slice real asset features
            real_features = real_features.reshape(-1, 1)
            real_features = real_features.to(x.device)  # ✅ fixed device assignment

            embedded = self.raw_projector(real_features)
            embedded = embedded.view(seq_len, num_assets + 1, self.embed_dim)
            embedded = embedded.mean(dim=1)  # average over assets

            out[i] = embedded

        if self.use_positional:
            out = self.positional_encoding(out)

        return out