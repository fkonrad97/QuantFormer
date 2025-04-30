import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces

df = load_all_iv_surfaces("data/real_world/iv_data/")
print(df[:5])

# Create buckets
df["strike_bucket"] = pd.qcut(df["strike"], q=4, labels=["low", "mid-low", "mid-high", "high"])
df["maturity_bucket"] = pd.qcut(df["maturity"], q=3, labels=["short", "medium", "long"])

# Plot IV, Strike, Maturity distributions
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df["iv"], bins=100, kde=True)
plt.title("IV Distribution")

plt.subplot(1, 3, 2)
sns.histplot(df["strike"], bins=100, kde=True)
plt.title("Strike Distribution")

plt.subplot(1, 3, 3)
sns.histplot(df["maturity"], bins=100, kde=True)
plt.title("Maturity Distribution")

plt.tight_layout()
plt.show()

# Boxplots for IV vs buckets
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="strike_bucket", y="iv")
plt.title("IV by Strike Bucket")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="maturity_bucket", y="iv")
plt.title("IV by Maturity Bucket")
plt.grid(True)
plt.show()

# Summary statistics
summary_stats = df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
print(summary_stats)