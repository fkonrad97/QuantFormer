import pandas as pd
import numpy as np
import torch
from data.synthetic.general_synthetic_iv_surface_dataset import GeneralSyntheticAssetIVSurfaceDataset
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces

def compare_synthetic_and_real_stats(real_data_folder="data/real_world/iv_data"):
    # === Load real-world data ===
    real_df = load_all_iv_surfaces(real_data_folder)
    print(f"Loaded {len(real_df)} real-world IV points.")

    # === Build synthetic dataset ===
    synthetic_ds = GeneralSyntheticAssetIVSurfaceDataset(
        min_assets=1,
        max_assets=1,
        num_strikes_per_asset=10,
        num_maturities=10,
        k_min=90,
        k_max=140,
        t_min=0.05,
        t_max=2.5,
        base_vol=np.random.uniform(0.30, 0.36),  # add surface-level variability
        skew_strength=-0.0008,                   # increase steepness a bit
        term_slope=0.002,                        # flatten further
        n_surfaces=200,
        device='cuda',
        randomize_skew=True,
        randomize_term=True,
        add_noise=True,
        noise_level=0.004                        # boost variability
    )


    # Flatten synthetic samples into one DataFrame
    synthetic_data = []
    for i in range(len(synthetic_ds)):
        features, ivs = synthetic_ds[i]
        for f, iv in zip(features, ivs):
            synthetic_data.append([f[0].item(), f[1].item(), iv.item()])  # strike, maturity, iv
    synthetic_df = pd.DataFrame(synthetic_data, columns=["strike", "maturity", "iv"])

    # === Compute summary statistics ===
    def summarize(df, name):
        print(f"\n{name} Stats:")
        for col in ['strike', 'maturity', 'iv']:
            print(f"  {col:>8}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}")
        print(f"  Corr(strike, iv):    {df[['strike', 'iv']].corr().iloc[0, 1]:.4f}")
        print(f"  Corr(maturity, iv):  {df[['maturity', 'iv']].corr().iloc[0, 1]:.4f}")

    summarize(synthetic_df, "Synthetic")
    summarize(real_df, "Real-World")

if __name__ == "__main__":
    compare_synthetic_and_real_stats("data/real_world/iv_data") 