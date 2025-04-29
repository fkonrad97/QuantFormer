import torch
from preprocessor.general_iv_surface_input_processor import GeneralInputProcessor
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor
from model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from data.synthetic.general_synthetic_iv_surface_dataset import GeneralSyntheticAssetIVSurfaceDataset, general_collate_fn
from data.real_world.real_iv_surface_dataset import RealIVSurfaceDataset
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces
import os

def test_general_iv_surface_transformer_model(use_masked=True):
    # Create a small synthetic dataset
    dataset = GeneralSyntheticAssetIVSurfaceDataset(
        min_assets=1,
        max_assets=2,
        num_strikes_per_asset=5,
        num_maturities=5,
        n_surfaces=3,
        device='cpu'
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=general_collate_fn)
    features, targets, asset_dims = next(iter(loader))

    # Model config
    embed_dim = 16
    model = GeneralIVSurfaceTransformerModel(
        input_processor=MaskedGeneralInputProcessor(
            input_dim=features.shape[-1],
            embed_dim=embed_dim,
            use_mlp=True
        ) if use_masked else GeneralInputProcessor(
            input_dim=features.shape[-1],
            embed_dim=embed_dim,
            use_mlp=True
        ),
        embed_dim=embed_dim,
        num_heads=2,
        ff_hidden_dim=32,
        num_layers=2,
        head_hidden_dim=16,
        dropout=0.1
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(features, asset_dims=asset_dims if use_masked else None, verbose=True)

    # Assertions
    assert preds.shape == targets.shape, f"Expected shape {targets.shape}, got {preds.shape}"
    print("Forward pass successful! Output shape matches target shape.")

def test_model_on_real_iv_data(real_data_folder="data/real_iv_surfaces"):
    if not os.path.exists(real_data_folder):
        print(f"Skipping real-world test: folder '{real_data_folder}' not found.")
        return

    try:
        df = load_all_iv_surfaces(real_data_folder)
        if df.empty:
            print("Real IV surface dataframe is empty.")
            return

        dataset = RealIVSurfaceDataset(df, normalize=True)

        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        # Build model
        embed_dim = 16
        model = GeneralIVSurfaceTransformerModel(
            input_processor=GeneralInputProcessor(input_dim=2, embed_dim=embed_dim, use_mlp=True),
            embed_dim=embed_dim,
            num_heads=2,
            ff_hidden_dim=32,
            num_layers=2,
            head_hidden_dim=16,
            dropout=0.1
        )

        model.eval()
        batch = next(iter(loader))
        features, targets = batch

        # Reshape to (batch_size=1, seq_len, input_dim)
        features = features.unsqueeze(0)
        targets = targets.unsqueeze(0)

        with torch.no_grad():
            preds = model(features, verbose=True)

        assert preds.shape == targets.shape, f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"
        print("Real-world IV surface forward pass successful!")

    except Exception as e:
        print(f"Error in real-world test: {e}")

if __name__ == "__main__":
    print("Running test with MaskedInputProcessor:")
    test_general_iv_surface_transformer_model(use_masked=True)
    
    print("\nRunning test with GeneralInputProcessor:")
    test_general_iv_surface_transformer_model(use_masked=False)

    print("\nRunning test with Real IV Surface Dataset:")
    test_model_on_real_iv_data(real_data_folder="data/real_world/iv_data") 