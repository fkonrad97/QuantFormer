from data.real_world.real_iv_surface_dataset import RealIVSurfaceDataset
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces

# Load data
df = load_all_iv_surfaces("data/real_world/iv_data", instruments=["MS"])  # point this to your real data
print("Loaded real data shape:", df.shape)

# Wrap in dataset
real_ds = RealIVSurfaceDataset(df, normalize=True)

# Fetch sample
x, y = real_ds[0]
print("Real Sample x:", x)  # Tensor([strike, maturity])
print("Real Sample y:", y)  # Tensor(implied_vol)

# Optional: inverse-transform and print
iv_original = real_ds.inverse_transform_iv(y)
print("Inverse Transformed IV:", iv_original.item())
