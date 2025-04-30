import pandas as pd
import numpy as np
import torch
from data.synthetic.logspace_synthetic_iv_surface_dataset import LogSpaceIVSurfaceDataset
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def compare_synthetic_and_real_stats(real_data_folder="data/real_world/iv_data"):
    # === Load real-world data ===
    real_df = load_all_iv_surfaces(real_data_folder)
    print(f"Loaded {len(real_df)} real-world IV points.")

    # === Build synthetic dataset ===
    synthetic_ds = LogSpaceIVSurfaceDataset(
        real_strikes=real_strikes,
        real_maturities=real_maturities,
        n_surfaces=500,
        num_points_per_surface=500,
        base_vol_range=(0.15, 0.25),
        skew_range=(-0.15, -0.05),
        term_slope_range=(-0.2, 0.05),
        noise_level=0.015,
        normalize=False
    )

    # === Flatten synthetic dataset into a DataFrame ===
    synthetic_data = []
    for i in range(len(synthetic_ds)):
        features, ivs = synthetic_ds[i]
        for f, iv in zip(features, ivs):
            strike = f[:-1].mean().item() if f.shape[0] > 1 else f[0].item()
            maturity = f[-1].item()
            synthetic_data.append([strike, maturity, iv.item()])

    synthetic_df = pd.DataFrame(synthetic_data, columns=["strike", "maturity", "iv"])

    # === Compute and print summary statistics ===
    def summarize(df, name):
        print(f"\n{name} Stats:")
        for col in ['strike', 'maturity', 'iv']:
            print(f"  {col:>8}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}")
        corr_strike_iv = df[['strike', 'iv']].corr().iloc[0, 1]
        corr_maturity_iv = df[['maturity', 'iv']].corr().iloc[0, 1]
        print(f"  Corr(strike, iv):    {corr_strike_iv:.4f}")
        print(f"  Corr(maturity, iv):  {corr_maturity_iv:.4f}")

    summarize(synthetic_df, "Synthetic")
    summarize(real_df, "Real-World")


def compare_iv_distributions(real_df, synthetic_dataset):
    """
    Compares real vs synthetic IV surfaces by plotting and printing descriptive stats.
    """
    # Flatten synthetic samples into DataFrame
    synthetic_data = []
    for i in range(len(synthetic_dataset)):
        features, ivs = synthetic_dataset[i]
        for f, iv in zip(features, ivs):
            strike = f[:-1].mean().item() if f.shape[0] > 1 else f[0].item()
            maturity = f[-1].item()
            synthetic_data.append([strike, maturity, iv.item()])

    synthetic_df = pd.DataFrame(synthetic_data, columns=["strike", "maturity", "iv"])

    def print_summary(name, df):
        print(f"\n{name} Stats:")
        for col in ["strike", "maturity", "iv"]:
            print(f"  {col:>8}: min={df[col].min():.4f}, max={df[col].max():.4f}, ",
                  f"mean={df[col].mean():.4f}, std={df[col].std():.4f}")
        print(f"  Corr(strike, iv):    {df[['strike', 'iv']].corr().iloc[0, 1]:.4f}")
        print(f"  Corr(maturity, iv):  {df[['maturity', 'iv']].corr().iloc[0, 1]:.4f}")

    print_summary("Real-World", real_df)
    print_summary("Synthetic", synthetic_df)

    # Plotting distributions
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    sns.histplot(real_df["iv"], bins=100, kde=True, ax=axs[0, 0], color="blue")
    axs[0, 0].set_title("Real IV Distribution")
    sns.histplot(real_df["strike"], bins=100, kde=True, ax=axs[0, 1], color="blue")
    axs[0, 1].set_title("Real Strike Distribution")
    sns.histplot(real_df["maturity"], bins=100, kde=True, ax=axs[0, 2], color="blue")
    axs[0, 2].set_title("Real Maturity Distribution")

    sns.histplot(synthetic_df["iv"], bins=100, kde=True, ax=axs[1, 0], color="orange")
    axs[1, 0].set_title("Synthetic IV Distribution")
    sns.histplot(synthetic_df["strike"], bins=100, kde=True, ax=axs[1, 1], color="orange")
    axs[1, 1].set_title("Synthetic Strike Distribution")
    sns.histplot(synthetic_df["maturity"], bins=100, kde=True, ax=axs[1, 2], color="orange")
    axs[1, 2].set_title("Synthetic Maturity Distribution")

    for ax in axs.flat:
        ax.grid(True)
        ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    return synthetic_df

if __name__ == "__main__":
    real_df = load_all_iv_surfaces("data/real_world/iv_data", instruments=['BLK','JPM','MS','GM', 'SPY'])
    real_strikes = real_df["strike"].values
    real_maturities = real_df["maturity"].values

    synthetic_ds = LogSpaceIVSurfaceDataset(
        real_strikes=real_strikes,
        real_maturities=real_maturities,
        n_surfaces=500,
        num_points_per_surface=500,
        normalize=False
    )

    compare_iv_distributions(real_df, synthetic_ds)