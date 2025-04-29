from torch.utils.data import DataLoader
from data.real_world.real_iv_multiasset_dataset import RealMultiAssetIVSurfaceDataset
from data.real_world.iv_surface_data_processor import build_iv_dataset

def test_multi_asset_iv_workflow():
    from torch.utils.data import DataLoader
    import torch
    from data.real_world.iv_surface_data_processor import build_iv_dataset
    from data.real_world.real_iv_multiasset_dataset import RealMultiAssetIVSurfaceDataset

    # Step 1: Build a multi-asset dataset (e.g., MS + JPM)
    df_multi = build_iv_dataset("data/real_world/iv_data", ["MS", "JPM"], method="mean")

    assert not df_multi.empty, "Generated multi-asset IV dataset is empty!"
    assert "strike_1" in df_multi.columns and "strike_2" in df_multi.columns, "Missing strike columns!"
    assert "iv" in df_multi.columns and "maturity" in df_multi.columns, "Missing required columns!"

    # Step 2: Check raw IV stats
    ivs_raw = torch.tensor(df_multi["iv"].values, dtype=torch.float32)
    print(f"[Raw IV Stats] min={ivs_raw.min():.4f}, max={ivs_raw.max():.4f}, mean={ivs_raw.mean():.4f}, std={ivs_raw.std():.4f}")
    assert ivs_raw.min() > 1e-3, "Raw IVs contain non-positive or near-zero values!"

    # Step 3: Load into dataset
    dataset = RealMultiAssetIVSurfaceDataset(df_multi, normalize=True)
    assert len(dataset) > 0, "Dataset has no entries!"

    # Step 4: Check normalized target stats
    all_y = torch.stack([dataset[i][1] for i in range(len(dataset))])
    print(f"[Normalized IV Stats] min={all_y.min():.2f}, max={all_y.max():.2f}")

    # Step 5: Check sample shapes
    x, y = dataset[0]
    assert x.ndim == 1, "x should be a 1D tensor"
    assert y.ndim == 0, "y should be a scalar"

    print("Sample feature:", x)
    print("Sample target:", y)

    # Step 6: Check DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_x, batch_y in loader:
        print("Batch x:", batch_x.shape)
        print("Batch y:", batch_y.shape)
        break  # one batch is enough

    print("Multi-asset IV dataset workflow test passed.")


test_multi_asset_iv_workflow()
