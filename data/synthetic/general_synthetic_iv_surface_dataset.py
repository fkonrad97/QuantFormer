import torch
from torch.utils.data import Dataset
import random
import numpy as np

class GeneralSyntheticAssetIVSurfaceDataset(Dataset):
    """
    Synthetic Dataset for General Multi-Asset Implied Volatility Surfaces.

    Generates synthetic implied volatility surfaces with configurable parameters,
    supporting multiple assets, randomization, and noise injection.

    Each sample corresponds to a full surface over strikes, maturities (and assets if multi-dimensional),
    providing features (strike, maturity, asset strikes) and corresponding volatility targets.

    Args:
        min_assets (int): Minimum number of assets per surface (default: 1).
        max_assets (int): Maximum number of assets per surface (default: 3).
        num_strikes_per_asset (int): Number of strike points per asset (default: 10).
        num_maturities (int): Number of maturity points (default: 10).
        k_min (float): Minimum strike value (default: 80).
        k_max (float): Maximum strike value (default: 120).
        t_min (float): Minimum maturity (default: 0.1 years).
        t_max (float): Maximum maturity (default: 2.0 years).
        base_vol (float): Base implied volatility level (default: 0.2).
        skew_strength (float): Base strength of volatility skew (default: 0.0005).
        term_slope (float): Base slope of volatility term structure (default: 0.05).
        n_surfaces (int): Number of surfaces to generate (default: 100).
        device (str): Device to place tensors on ('cpu' or 'cuda') (default: 'cpu').
        randomize_skew (bool): If True, randomizes skew strength per surface (default: False).
        randomize_term (bool): If True, randomizes term slope per surface (default: False).
        add_noise (bool): If True, adds Gaussian noise to volatilities (default: False).
        noise_level (float): Standard deviation of Gaussian noise to inject (default: 0.005).

    Returns:
        features (torch.Tensor): Shape (num_points, num_features) per surface.
        targets (torch.Tensor): Shape (num_points,) corresponding volatilities.
    """
    def __init__(
        self,
        min_assets=1,
        max_assets=3,
        num_strikes_per_asset=10,
        num_maturities=10,
        k_min=80,
        k_max=120,
        t_min=0.1,
        t_max=2.0,
        base_vol_range=(0.2, 0.4),
        skew_strength_range=(-0.004, -0.0005),
        term_slope_range=(-0.015, 0.015),
        n_surfaces=100,
        device='cpu',
        randomize_skew=False,
        randomize_term=False,
        add_noise=False,
        noise_level=0.005,
        atm_jitter_fraction=0.25,
        normalize=False
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.n_surfaces = n_surfaces
        self.normalize = normalize

        self.randomize_skew = randomize_skew
        self.randomize_term = randomize_term
        self.add_noise = add_noise
        self.noise_level = noise_level

        self.k_min = k_min
        self.k_max = k_max
        self.t_min = t_min
        self.t_max = t_max
        self.num_strikes_per_asset = num_strikes_per_asset
        self.num_maturities = num_maturities
        self.min_assets = min_assets
        self.max_assets = max_assets
        self.atm_jitter_fraction = atm_jitter_fraction

        self.base_vol_range = base_vol_range
        self.skew_strength_range = skew_strength_range
        self.term_slope_range = term_slope_range

        surfaces = []
        all_strikes = []
        all_maturities = []
        all_ivs = []

        for _ in range(n_surfaces):
            num_assets = random.randint(min_assets, max_assets)
            features, target = self.generate_single_surface(num_assets)
            surfaces.append((features, target))

            all_strikes.append(features[:, :-1].reshape(-1))
            all_maturities.append(features[:, -1])
            all_ivs.append(target)

        self.strikes = torch.cat(all_strikes).view(-1)
        self.maturities = torch.cat(all_maturities).view(-1)
        self.ivs = torch.cat(all_ivs).view(-1)

        if normalize:
            self.strike_mean = self.strikes.mean()
            self.strike_std = self.strikes.std()
            self.maturity_mean = self.maturities.mean()
            self.maturity_std = self.maturities.std()
            self.iv_mean = self.ivs.mean()
            self.iv_std = self.ivs.std()

            normalized_surfaces = []
            for features, target in surfaces:
                for i in range(features.shape[1] - 1):
                    features[:, i] = (features[:, i] - self.strike_mean) / self.strike_std
                features[:, -1] = (features[:, -1] - self.maturity_mean) / self.maturity_std
                target = (target - self.iv_mean) / self.iv_std
                normalized_surfaces.append((features, target))
            self.surfaces = normalized_surfaces
        else:
            self.surfaces = surfaces

    def generate_single_surface(self, num_assets):
        """
        Generates a synthetic implied volatility surface with realistic skew,
        term structure, and a skewed base_vol distribution.
        """
        strike_grids = [
            torch.linspace(self.k_min, self.k_max, self.num_strikes_per_asset, device=self.device)
            for _ in range(num_assets)
        ]
        T = torch.linspace(self.t_min, self.t_max, self.num_maturities, device=self.device)
        meshes = torch.meshgrid(*strike_grids, T, indexing='ij')
        coords = torch.stack([g.flatten() for g in meshes], dim=-1)

        # === Sample base_vol from a skewed distribution (Beta) ===
        a, b = 2, 5  # shape params: skewed toward low vol
        base_vol = np.random.beta(a, b) * (self.base_vol_range[1] - self.base_vol_range[0]) + self.base_vol_range[0]

        # === Randomize skew and term slope ===
        skew = (
            np.random.uniform(*self.skew_strength_range)
            if self.randomize_skew else self.skew_strength_range[0]
        )
        term_slope = (
            np.random.uniform(*self.term_slope_range)
            if self.randomize_term else self.term_slope_range[0]
        )

        # === ATM reference level ===
        atm_low = self.k_min + self.atm_jitter_fraction * (self.k_max - self.k_min)
        atm_high = self.k_max - self.atm_jitter_fraction * (self.k_max - self.k_min)
        K_atm = torch.FloatTensor(1).uniform_(atm_low, atm_high).item()

        # === Build vol surface ===
        vol_surface = base_vol
        for i in range(num_assets):
            K = meshes[i]
            vol_surface += skew * (K - K_atm)

        T_mesh = meshes[-1]
        vol_surface += term_slope * T_mesh

        # === Add beta-distributed noise ===
        if self.add_noise:
            beta_noise = np.random.beta(2, 5, size=vol_surface.shape) * (self.noise_level / 2)
            beta_noise = torch.tensor(beta_noise, dtype=torch.float32, device=self.device)
            vol_surface += beta_noise

        # clamp
        # vol_surface = torch.clamp(vol_surface, min=0.05, max=1.1)

        return coords, vol_surface.flatten()

    
    def __len__(self):
        """
        Returns:
            int: Number of surfaces in the dataset.
        """
        return len(self.surfaces)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the surface to fetch.
    
        Returns:
            tuple: (features, targets) tensors for one surface.
        """
        return self.surfaces[idx]



def general_collate_fn(batch):
    """
    Collate function for batching variable asset IV surfaces.

    This collate function is designed for datasets where each sample may have a different
    number of assets (i.e., different input feature dimensions).
    It dynamically pads features and targets so that they can be batched together
    into uniform tensors for training a generalizable model.

    Args:
        batch (list of tuples): A batch of samples, where each sample is a tuple:
            - features (torch.Tensor): shape (num_points, input_dim),
              input_dim = num_assets + 1 (maturity included)
            - targets (torch.Tensor): shape (num_points,), corresponding volatilities

    Returns:
        padded_features (torch.Tensor): shape (batch_size, max_num_points, max_input_dim)
            - Features padded along feature dimension (number of strike inputs) and point dimension (number of grid points).
            - Feature padding is done with zeros.
        padded_targets (torch.Tensor): shape (batch_size, max_num_points)
            - Targets padded along number of points with zeros.
        asset_dims (list of int): List of asset counts for each sample in the batch.
            - For each sample, asset_dim = input_dim - 1 (maturity is not counted as an asset).
    
    Notes:
        - Features are padded to match the maximum input dimension (max number of assets + 1 maturity) across the batch.
        - Number of points (grid size) is also padded to the maximum among batch samples.
        - Padding is done with zeros for both features and targets.
        - The returned asset_dims list can be used later to reconstruct real asset structures or apply masks if needed.
    """
    features_list, targets_list = zip(*batch)

    # Find maximum input dimension (assets + maturity) and maximum num_points
    max_input_dim = max(f.shape[1] for f in features_list)
    max_num_points = max(f.shape[0] for f in features_list)

    padded_features = []
    padded_targets = []
    asset_dims = []

    for features, targets in zip(features_list, targets_list):
        num_points, input_dim = features.shape
        asset_dims.append(input_dim - 1)  # Exclude maturity

        # Pad feature dimension if needed
        feature_pad_size = max_input_dim - input_dim
        if feature_pad_size > 0:
            features = torch.cat([features, torch.zeros(num_points, feature_pad_size, device=features.device)], dim=-1)

        # Pad number of points if needed
        if num_points < max_num_points:
            feature_pad = torch.zeros(max_num_points - num_points, max_input_dim, device=features.device)
            features = torch.cat([features, feature_pad], dim=0)

            target_pad = torch.zeros(max_num_points - num_points, device=targets.device)
            targets = torch.cat([targets, target_pad], dim=0)

        padded_features.append(features)
        padded_targets.append(targets)

    padded_features = torch.stack(padded_features, dim=0)
    padded_targets = torch.stack(padded_targets, dim=0)

    return padded_features, padded_targets, asset_dims
