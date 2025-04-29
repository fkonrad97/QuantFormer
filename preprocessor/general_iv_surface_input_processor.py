import torch
import torch.nn as nn
from embeddings.tabular_embeddings import TabularEmbedding
from embeddings.positional_encoding import PositionalEncoding

class GeneralInputProcessor(nn.Module):
    """
    General Input Processor for variable asset IV surfaces.

    This module processes batched tabular inputs where the number of assets
    may vary across samples but have been padded to a consistent input dimension.
    It embeds the entire input tensor (including any padded zeros) using a tabular
    embedding (Linear or shallow MLP projection) and optionally applies
    positional encoding to the resulting embeddings.

    Args:
        input_dim (int): Total input dimension (max number of assets + maturity).
        embed_dim (int): Output embedding dimension for the Transformer.
        use_mlp (bool, optional): If True, use a shallow MLP (2-layer) instead of a single Linear projection. Default: False.
        use_positional (bool, optional): If True, add sinusoidal positional encodings to the embeddings. Default: True.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

    Returns:
        out (torch.Tensor): Output embedded tensor of shape (batch_size, seq_len, embed_dim).

    Notes:
        - Padded zero features are projected along with real features.
        - Model is expected to learn that padded dimensions are uninformative during training.
        - This is a simple and robust strategy for real-world tabular + sequence modeling.
    """
    def __init__(self, input_dim, embed_dim, use_mlp=False, use_positional=True):
        super().__init__()
        self.tabular_embedding = TabularEmbedding(input_dim=input_dim, embed_dim=embed_dim, use_mlp=use_mlp)
        self.use_positional = use_positional

        if self.use_positional:
            self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        assert x.ndim == 3, f"Expected input of shape (B, T, D), got {x.shape}"
        x = self.tabular_embedding(x)
        if self.use_positional:
            x = self.positional_encoding(x)
        return x