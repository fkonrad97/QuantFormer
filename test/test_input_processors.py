import torch
from preprocessor.general_iv_surface_input_processor import GeneralInputProcessor
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor

def test_general_input_processor():
    batch_size = 2
    seq_len = 5
    input_dim = 4  # e.g., 3 assets + 1 maturity
    embed_dim = 8

    x = torch.randn(batch_size, seq_len, input_dim)
    processor = GeneralInputProcessor(input_dim=input_dim, embed_dim=embed_dim, use_mlp=True, use_positional=True)
    out = processor(x)

    assert out.shape == (batch_size, seq_len, embed_dim), f"Unexpected shape: {out.shape}"
    print("GeneralInputProcessor passed!")

def test_masked_input_processor():
    batch_size = 2
    seq_len = 5
    max_input_dim = 5  # e.g., max 4 assets + 1 maturity
    embed_dim = 8

    # Variable asset counts per sample
    asset_dims = [2, 3]  # sample 0 has 2 assets, sample 1 has 3 assets

    # Create dummy input padded with zeros beyond real asset dims
    x = torch.zeros(batch_size, seq_len, max_input_dim)
    x[0, :, :3] = torch.randn(seq_len, 3)  # 2 assets + 1 maturity
    x[1, :, :4] = torch.randn(seq_len, 4)  # 3 assets + 1 maturity

    processor = MaskedGeneralInputProcessor(input_dim=max_input_dim, embed_dim=embed_dim, use_mlp=True, use_positional=True)
    out = processor(x, asset_dims)

    assert out.shape == (batch_size, seq_len, embed_dim), f"Unexpected shape: {out.shape}"
    print("MaskedGeneralInputProcessor passed!")

if __name__ == "__main__":
    test_general_input_processor()
    test_masked_input_processor()
