import torch
from torch.utils.data import Dataset

class RealMultiAssetIVSurfaceDataset(Dataset):
    """
    PyTorch Dataset for multi-asset (or single-asset) real-world implied volatility surface data.
    Supports dynamic number of strike inputs (e.g., strike_1, strike_2, ..., strike_N, maturity → iv).
    """

    def __init__(self, dataframe, normalize=True):
        """
        Args:
            dataframe (pd.DataFrame): Must contain 'maturity', 'iv', and one or more 'strike_*' columns.
            normalize (bool): Whether to normalize features and targets.
        """
        self.normalize = normalize

        # Detect all strike_* columns
        self.strike_cols = [col for col in dataframe.columns if col.startswith("strike")]
        self.maturity_col = "maturity"
        self.iv_col = "iv"

        self.features = dataframe[self.strike_cols + [self.maturity_col]].values.astype("float32")
        self.targets = dataframe[self.iv_col].values.astype("float32")

        self.features = torch.tensor(self.features)
        self.targets = torch.tensor(self.targets)

        if normalize:
            print(f"[Raw IV Stats] min={self.targets.min():.4f}, max={self.targets.max():.4f}, "
              f"mean={self.targets.mean():.4f}, std={self.targets.std():.4f}")
            self.feature_mean = self.features.mean(dim=0)
            self.feature_std = self.features.std(dim=0)
            self.iv_mean = self.targets.mean()
            self.iv_std = self.targets.std()

            self.features = (self.features - self.feature_mean) / self.feature_std
            self.targets = (self.targets - self.iv_mean) / self.iv_std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y

    def inverse_transform_iv(self, iv_tensor):
        if self.normalize:
            return iv_tensor * self.iv_std + self.iv_mean
        return iv_tensor

    def inverse_transform_features(self, feature_tensor):
        if self.normalize:
            return feature_tensor * self.feature_std + self.feature_mean
        return feature_tensor