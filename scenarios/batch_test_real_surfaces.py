import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor

# === Config ===
data_dir = "data/real_world/iv_data/MS/"  # folder containing *_iv_surface.csv
model_path = "finetuned_multiasset_model.pth"

# === Load Model ===
input_processor = MaskedGeneralInputProcessor(input_dim=2, embed_dim=64, use_mlp=True)
model = GeneralIVSurfaceTransformerModel(
    input_processor=input_processor,
    embed_dim=64,
    num_heads=4,
    ff_hidden_dim=128,
    num_layers=4,
    head_hidden_dim=64,
    dropout=0.1
)
model.load_state_dict(torch.load(model_path))
model.eval()

# === Process All Surfaces ===
csv_files = sorted(glob(os.path.join(data_dir, "*_iv_surface.csv")))
results = []

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
        features = df[["strike", "maturity"]].values.astype("float32")
        targets = df["iv"].values.astype("float32")

        x = torch.tensor(features).unsqueeze(0)  # [1, N, 2]
        y = torch.tensor(targets).unsqueeze(0)   # [1, N]
        asset_dims = [x.shape[-1] - 1]

        with torch.no_grad():
            pred = model(x, asset_dims=asset_dims).squeeze(0)

        mae = mean_absolute_error(y.squeeze(0), pred)
        mse = mean_squared_error(y.squeeze(0), pred)
        r2 = r2_score(y.squeeze(0), pred)
        mape = (torch.abs(pred - y.squeeze(0)) / y.squeeze(0)).mean().item() * 100

        results.append({
            "surface": os.path.basename(csv_path),
            "MAE": mae,
            "MSE": mse,
            "MAPE": mape,
            "R2": r2
        })

    except Exception as e:
        print(f"❌ Failed on {csv_path}: {e}")

# === Output Results ===
results_df = pd.DataFrame(results)
print("\n=== Surface-by-Surface Evaluation ===")
print(results_df)

print("\n=== Averages Across All Surfaces ===")
print(f"MAE:  {results_df['MAE'].mean():.4f}")
print(f"MSE:  {results_df['MSE'].mean():.6f}")
print(f"MAPE: {results_df['MAPE'].mean():.2f}%")
print(f"R²:   {results_df['R2'].mean():.4f}")
