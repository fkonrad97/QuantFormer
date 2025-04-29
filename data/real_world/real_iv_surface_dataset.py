import torch
from torch.utils.data import Dataset

class RealIVSurfaceDataset(Dataset):
    """
    PyTorch Dataset for real-world implied volatility surface data.
    Each sample is (strike, maturity) -> implied volatility.
    """
    def __init__(self, dataframe, normalize=False):
        """
        Args:
            dataframe (pd.DataFrame): Contains 'strike', 'maturity', 'iv'.
            normalize (bool): Whether to apply normalization.
        """
        self.normalize = normalize

        self.strikes = torch.tensor(dataframe['strike'].values, dtype=torch.float32)
        self.maturities = torch.tensor(dataframe['maturity'].values, dtype=torch.float32)
        self.ivs = torch.tensor(dataframe['iv'].values, dtype=torch.float32)

        if normalize:
            self.strike_mean = self.strikes.mean()
            self.strike_std = self.strikes.std()
            self.maturity_mean = self.maturities.mean()
            self.maturity_std = self.maturities.std()
            self.iv_mean = self.ivs.mean()
            self.iv_std = self.ivs.std()

            self.strikes = (self.strikes - self.strike_mean) / self.strike_std
            self.maturities = (self.maturities - self.maturity_mean) / self.maturity_std
            self.ivs = (self.ivs - self.iv_mean) / self.iv_std

    def __len__(self):
        return len(self.ivs)

    def __getitem__(self, idx):
        x = torch.tensor([self.strikes[idx], self.maturities[idx]], dtype=torch.float32)
        y = self.ivs[idx]
        return x, y
    
    def inverse_transform_iv(self, iv_tensor):
        if self.normalize:
            return iv_tensor * self.iv_std + self.iv_mean
        return iv_tensor