import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import gaussian_kde

class LearnableIVSurfaceGenerator(nn.Module):
    """
    A learnable synthetic IV surface generator in log-space.

    IV(K, T) = exp(log(base_vol) + skew * (K - K_atm) + slope * T + eps)

    Supports batched surface generation: one parameter set per surface in batch.
    """
    def __init__(self, num_assets, device="cpu", init_ranges=None, add_noise=True, noise_std=0.02):
        super().__init__()
        self.num_assets = num_assets
        self.device = torch.device(device)
        self.add_noise = add_noise
        self.noise_std = noise_std

        # Parameter ranges for initialization
        self.init_ranges = init_ranges or {
            "base_vol": (0.2, 0.5),
            "skew": (-0.1, -0.01),
            "slope": (-0.1, 0.1)
        }

        # Register per-surface learnable parameters (batch size handled during forward)
        self.base_vol = nn.Parameter(torch.empty(1))
        self.skew = nn.Parameter(torch.empty(num_assets))
        self.term_slope = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        base_min, base_max = self.init_ranges["base_vol"]
        skew_min, skew_max = self.init_ranges["skew"]
        slope_min, slope_max = self.init_ranges["slope"]

        nn.init.uniform_(self.base_vol, base_min, base_max)
        nn.init.uniform_(self.skew, skew_min, skew_max)
        nn.init.uniform_(self.term_slope, slope_min, slope_max)

    def forward(self, coords: torch.Tensor, K_atm: float = None):
        B, N, D = coords.shape
        assert D == self.num_assets + 1, "Mismatch between input and asset count"

        coords = coords.to(self.device)  # ensure input is on same device
        strikes = coords[:, :, :-1]
        maturities = coords[:, :, -1]

        if K_atm is None:
            K_min = strikes.min(dim=1, keepdim=True).values
            K_max = strikes.max(dim=1, keepdim=True).values
            K_atm = (K_min + K_max) / 2

        # Also ensure K_atm is on the same device
        K_atm = K_atm.to(self.device)

        strike_shift = (strikes - K_atm)
        skew_term = (strike_shift * self.skew.view(1, 1, -1)).sum(dim=-1)

        time_term = maturities * self.term_slope

        log_iv = torch.log(self.base_vol) + skew_term + time_term

        if self.add_noise:
            log_iv += self.noise_std * torch.randn_like(log_iv)

        return torch.exp(log_iv)

class IVSurfaceSampler:
    def __init__(self, real_strikes, real_maturities):
        self.strike_kde = gaussian_kde(real_strikes)
        self.maturity_kde = gaussian_kde(real_maturities)

    def sample(self, num_samples, num_assets):
        while True:
            strikes = [self.strike_kde.resample(num_samples).flatten() for _ in range(num_assets)]
            maturities = self.maturity_kde.resample(num_samples).flatten()
            if np.all([np.all(k > 0) for k in strikes]) and np.all(maturities > 0):
                coords = np.stack(strikes + [maturities], axis=1)
                return torch.tensor(coords, dtype=torch.float32)

class SyntheticIVDataset(Dataset):
    def __init__(self, generator: LearnableIVSurfaceGenerator, sampler: IVSurfaceSampler, 
                 num_surfaces=100, num_points=500):
        self.generator = generator
        self.sampler = sampler
        self.num_surfaces = num_surfaces
        self.num_points = num_points
        self.num_assets = generator.num_assets

    def __len__(self):
        return self.num_surfaces

    def __getitem__(self, idx):
        coords = self.sampler.sample(self.num_points, self.num_assets)  # (N, D)
        coords = coords.unsqueeze(0)  # (1, N, D)
        with torch.no_grad():
            ivs = self.generator(coords).squeeze(0)
        return coords.squeeze(0), ivs

if __name__ == "__main__":
    # Example usage with toy data
    real_strikes = np.random.uniform(80, 120, size=1000)
    real_maturities = np.random.uniform(0.1, 2.0, size=1000)

    generator = LearnableIVSurfaceGenerator(num_assets=2, device="cpu")
    sampler = IVSurfaceSampler(real_strikes, real_maturities)
    dataset = SyntheticIVDataset(generator, sampler)

    coords, ivs = dataset[0]
    print("Sample coords shape:", coords.shape)
    print("Sample IVs shape:", ivs.shape)
    print("IVs (first 5):", ivs[:5])