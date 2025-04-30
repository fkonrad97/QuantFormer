import torch
from torch.utils.data import Dataset
import numpy as np
import random
from scipy.stats import gaussian_kde

class LogSpaceIVSurfaceDataset(Dataset):
    def __init__(
        self,
        real_strikes,
        real_maturities,
        n_surfaces=100,
        min_assets=1,
        max_assets=3,
        num_points_per_surface=500,
        base_vol_range=(0.18, 0.28),
        skew_range=(-0.15, -0.05),
        term_slope_range=(-0.15, 0.03),
        noise_level=0.025,
        add_noise=True,
        atm_jitter_fraction=0.25,
        normalize=False,
        device="cpu"
    ):
        super().__init__()
        self.n_surfaces = n_surfaces
        self.min_assets = min_assets
        self.max_assets = max_assets
        self.num_points_per_surface = num_points_per_surface

        self.strike_kde = gaussian_kde(real_strikes)
        self.maturity_kde = gaussian_kde(real_maturities)

        self.base_vol_range = base_vol_range
        self.skew_range = skew_range
        self.term_slope_range = term_slope_range
        self.noise_level = noise_level
        self.add_noise = add_noise
        self.atm_jitter_fraction = atm_jitter_fraction
        self.normalize = normalize
        self.device = torch.device(device)

        self.surfaces = []
        self._build_dataset()

    def sample_strike(self, n=1):
        while True:
            samples = self.strike_kde.resample(n).flatten()
            if np.all(samples > 0) and np.all(samples < 1200):
                return samples if n > 1 else samples[0]

    def sample_maturity(self, n=1):
        while True:
            samples = self.maturity_kde.resample(n).flatten()
            if np.all(samples > 0) and np.all(samples < 5.0):
                return samples if n > 1 else samples[0]

    def _build_dataset(self):
        strikes_all = []
        maturities_all = []
        ivs_all = []

        for _ in range(self.n_surfaces):
            num_assets = random.randint(self.min_assets, self.max_assets)
            features, ivs = self._generate_surface(num_assets)
            self.surfaces.append((features, ivs))

            strikes_all.append(features[:, :-1].reshape(-1))
            maturities_all.append(features[:, -1])
            ivs_all.append(ivs)

        if self.normalize:
            self.strike_mean = torch.cat(strikes_all).mean()
            self.strike_std = torch.cat(strikes_all).std()
            self.maturity_mean = torch.cat(maturities_all).mean()
            self.maturity_std = torch.cat(maturities_all).std()
            self.iv_mean = torch.cat(ivs_all).mean()
            self.iv_std = torch.cat(ivs_all).std()

            normalized = []
            for features, target in self.surfaces:
                for i in range(features.shape[1] - 1):
                    features[:, i] = (features[:, i] - self.strike_mean) / self.strike_std
                features[:, -1] = (features[:, -1] - self.maturity_mean) / self.maturity_std
                target = (target - self.iv_mean) / self.iv_std
                normalized.append((features, target))
            self.surfaces = normalized

    def _generate_surface(self, num_assets):
        strikes = [torch.tensor(self.sample_strike(self.num_points_per_surface), device=self.device) for _ in range(num_assets)]
        maturities = torch.tensor(self.sample_maturity(self.num_points_per_surface), device=self.device)
        coords = torch.stack(strikes + [maturities], dim=1)

        base_vol = np.random.uniform(*self.base_vol_range)
        skew = np.random.uniform(*self.skew_range)
        term_slope = np.random.uniform(*self.term_slope_range)

        atm_low = torch.min(coords[:, :-1]) + self.atm_jitter_fraction * (torch.max(coords[:, :-1]) - torch.min(coords[:, :-1]))
        atm_high = torch.max(coords[:, :-1]) - self.atm_jitter_fraction * (torch.max(coords[:, :-1]) - torch.min(coords[:, :-1]))
        K_atm = torch.FloatTensor(1).uniform_(atm_low.item(), atm_high.item()).item()

        log_iv = np.log(base_vol)

        for i in range(num_assets):
            strike_shift = (coords[:, i] - K_atm) / (torch.max(coords[:, i]) - torch.min(coords[:, i]))
            log_iv += skew * strike_shift

        time_shift = (coords[:, -1] - torch.min(coords[:, -1])) / (torch.max(coords[:, -1]) - torch.min(coords[:, -1]))
        log_iv += term_slope * time_shift

        if self.add_noise:
            log_iv += self.noise_level * torch.randn_like(log_iv)

        iv_surface = torch.exp(log_iv)
        return coords, iv_surface.flatten()

    def __len__(self):
        return len(self.surfaces)

    def __getitem__(self, idx):
        return self.surfaces[idx]

    def inverse_transform_iv(self, iv_tensor):
        if self.normalize:
            return iv_tensor * self.iv_std + self.iv_mean
        return iv_tensor