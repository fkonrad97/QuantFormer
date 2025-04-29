import torch
from blocks.transformer_block import TransformerBlock
from encoder.transformer_encoder import TransformerEncoder

def test_transformer_block():
    batch_size = 2
    seq_len = 10
    embed_dim = 16
    ff_hidden_dim = 32
    num_heads = 4

    x = torch.randn(batch_size, seq_len, embed_dim)

    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim)
    out = block(x)

    assert out.shape == (batch_size, seq_len, embed_dim), f"Unexpected output shape from TransformerBlock: {out.shape}"
    print("TransformerBlock forward pass successful!")

def test_transformer_encoder():
    batch_size = 2
    seq_len = 10
    embed_dim = 16
    ff_hidden_dim = 32
    num_heads = 4
    num_layers = 3

    x = torch.randn(batch_size, seq_len, embed_dim)

    encoder = TransformerEncoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers
    )

    out = encoder(x)

    assert out.shape == (batch_size, seq_len, embed_dim), f"Unexpected output shape from TransformerEncoder: {out.shape}"
    print("TransformerEncoder forward pass successful!")

if __name__ == "__main__":
    print("Running TransformerBlock test:")
    test_transformer_block()

    print("\nRunning TransformerEncoder test:")
    test_transformer_encoder()