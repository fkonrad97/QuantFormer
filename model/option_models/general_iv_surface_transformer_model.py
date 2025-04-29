import torch
import torch.nn as nn
from model.modular_transformer_model import ModularTransformerModel
from encoder.transformer_encoder import TransformerEncoder
from heads.option_vol_surface_head import VolSurfaceHead

'''
x_batch + asset_dims
    ↓
input_processor (GeneralInputProcessor or MaskedGeneralInputProcessor)
    ↓
TransformerEncoder
    ↓
VolSurfaceHead
    ↓
Predicted Vol Surface
'''

class GeneralIVSurfaceTransformerModel(ModularTransformerModel):
    """
    General Transformer Model for Multi-Asset Volatility Surface Prediction.

    Supports any number of assets and maturities without assuming 2D grid structure.
    """

    def __init__(
        self,
        input_processor: nn.Module,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_hidden_dim=ff_hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        head = VolSurfaceHead(
            embed_dim=embed_dim,
            hidden_dim=head_hidden_dim,
        )

        super().__init__(
            input_processor=input_processor,
            encoder=encoder,
            head=head
        )

    def forward(self, x, asset_dims=None, mask=None, verbose=False):
        if asset_dims is not None:
            embedded = self.input_processor(x, asset_dims)
        else:
            embedded = self.input_processor(x)

        if verbose: 
            print(f"[InputProcessor] embedded shape: {embedded.shape}")
        encoded = self.encoder(embedded, mask)

        if verbose: 
            print(f"[TransformerEncoder] encoded shape: {encoded.shape}")
        out = self.head(encoded)

        if verbose: print(f"[VolSurfaceHead] output shape: {out.shape}")
        return out
