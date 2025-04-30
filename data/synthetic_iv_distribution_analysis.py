from data.synthetic.general_synthetic_iv_surface_dataset import GeneralSyntheticAssetIVSurfaceDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data.synthetic.logspace_synthetic_iv_surface_dataset import LogSpaceIVSurfaceDataset

# === Build synthetic dataset ===
# Instantiate the dataset
synthetic_ds = LogSpaceIVSurfaceDataset(
    n_surfaces=500,
    num_points_per_surface=500,
    normalize=False
)

synthetic_data = []
for i in range(len(synthetic_ds)):
    features, targets = synthetic_ds[i]
    for f, iv in zip(features, targets):
        synthetic_data.append([f[0].item(), f[1].item(), iv.item()])

synthetic_df = pd.DataFrame(synthetic_data, columns=["strike", "maturity", "iv"])

# Create buckets
synthetic_df["strike_bucket"] = pd.qcut(synthetic_df["strike"], q=4, labels=["low", "mid-low", "mid-high", "high"])
synthetic_df["maturity_bucket"] = pd.qcut(synthetic_df["maturity"], q=3, labels=["short", "medium", "long"])

# Plot distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(synthetic_df["iv"], bins=100, kde=True)
plt.title("Synthetic IV Distribution")

plt.subplot(1, 3, 2)
sns.histplot(synthetic_df["strike"], bins=100, kde=True)
plt.title("Synthetic Strike Distribution")

plt.subplot(1, 3, 3)
sns.histplot(synthetic_df["maturity"], bins=100, kde=True)
plt.title("Synthetic Maturity Distribution")

plt.tight_layout()
plt.show()

# Boxplots for IV vs strike and maturity buckets
plt.figure(figsize=(10, 6))
sns.boxplot(data=synthetic_df, x="strike_bucket", y="iv")
plt.title("Synthetic IV by Strike Bucket")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=synthetic_df, x="maturity_bucket", y="iv")
plt.title("Synthetic IV by Maturity Bucket")
plt.grid(True)
plt.show()

# Summary statistics
synthetic_summary = synthetic_df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
print(synthetic_summary)