import torch
import numpy as np
import matplotlib.pyplot as plt

# Configuration
num_surfaces = 500
num_strikes = 20
num_maturities = 20

k_min, k_max = 80, 120
t_min, t_max = 0.1, 2.0

base_vol_range = (0.2, 0.35)
skew_range = (-0.006, -0.001)
term_slope_range = (-0.02, 0.03)
noise_level = 0.004
atm_jitter_fraction = 0.25
add_noise = True

all_iv_values = []

for _ in range(num_surfaces):
    K = torch.linspace(k_min, k_max, num_strikes)
    T = torch.linspace(t_min, t_max, num_maturities)
    KK, TT = torch.meshgrid(K, T, indexing="ij")

    # Randomize base parameters per surface
    base_vol = np.random.uniform(*base_vol_range)
    skew = np.random.uniform(*skew_range)
    term_slope = np.random.uniform(*term_slope_range)

    atm_low = k_min + atm_jitter_fraction * (k_max - k_min)
    atm_high = k_max - atm_jitter_fraction * (k_max - k_min)
    K_atm = torch.FloatTensor(1).uniform_(atm_low, atm_high).item()

    log_iv = np.log(base_vol)
    log_iv += skew * (KK - K_atm)
    log_iv += term_slope * TT

    if add_noise:
        log_iv += noise_level * torch.randn_like(KK)

    iv_surface = torch.exp(log_iv)
    all_iv_values.append(iv_surface.flatten())

# Concatenate all values and compute statistics
all_iv_values = torch.cat(all_iv_values)

print(f"Summary statistics over {num_surfaces} surfaces:")
print(f"  Min IV:   {all_iv_values.min().item():.4f}")
print(f"  Max IV:   {all_iv_values.max().item():.4f}")
print(f"  Mean IV:  {all_iv_values.mean().item():.4f}")
print(f"  Std IV:   {all_iv_values.std().item():.4f}")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_iv_values.numpy(), bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Synthetic IVs (Log-Space Model)")
plt.xlabel("Implied Volatility")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()