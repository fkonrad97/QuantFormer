from data.synthetic.general_synthetic_iv_surface_dataset import GeneralSyntheticAssetIVSurfaceDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plot

# Create a synthetic dataset
synthetic_ds = GeneralSyntheticAssetIVSurfaceDataset(
    min_assets=1,
    max_assets=1,
    num_strikes_per_asset=10,
    num_maturities=10,
    n_surfaces=5,
    device='cuda',  # will auto-fallback if unavailable
    add_noise=True,
    randomize_skew=True,
    randomize_term=True,
    normalize=True
)

# Print shape info
features, targets = synthetic_ds[0]
print("Synthetic Sample Features:", features.shape)  # should be (100, 2)
print("Synthetic Sample Targets:", targets.shape)    # should be (100,)
print("Device:", features.device)

# Optional: Plot the surface
strikes = features[:, 0].cpu().numpy().reshape(10, 10)
maturities = features[:, 1].cpu().numpy().reshape(10, 10)
vols = targets.cpu().numpy().reshape(10, 10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(strikes, maturities, vols, cmap='viridis')
ax.set_title("Synthetic IV Surface")
plt.show()
