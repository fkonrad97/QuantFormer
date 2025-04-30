import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# === Load Surface CSV ===
df = pd.read_csv("data/real_world/iv_data/GM/GM_2025-04-01_iv_surface.csv")
features = df[["strike", "maturity"]].values.astype("float32")
targets = df["iv"].values.astype("float32")

x = torch.tensor(features).unsqueeze(0)  # [1, N, 2]
y = torch.tensor(targets).unsqueeze(0)   # [1, N]
asset_dims = [x.shape[-1] - 1]

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
model.load_state_dict(torch.load("finetuned_multiasset_model.pth"))
model.eval()

# === Predict ===
with torch.no_grad():
    pred = model(x, asset_dims=asset_dims).squeeze(0)  # [N]

# === Metrics ===
mae = mean_absolute_error(y.squeeze(0), pred)
mse = mean_squared_error(y.squeeze(0), pred)
r2 = r2_score(y.squeeze(0), pred)
mape = (torch.abs(pred - y.squeeze(0)) / y.squeeze(0)).mean().item() * 100

print(f"\nEvaluation on GM 2025-04-01 Surface:")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.6f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

# === Plot Heatmaps ===
df["pred_iv"] = pred.numpy()

pivot_true = df.pivot(index="strike", columns="maturity", values="iv")
pivot_pred = df.pivot(index="strike", columns="maturity", values="pred_iv")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im0 = axs[0].imshow(pivot_true.values, cmap="viridis", aspect="auto", origin="lower")
axs[0].set_title("True IV Surface")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(pivot_pred.values, cmap="viridis", aspect="auto", origin="lower")
axs[1].set_title("Predicted IV Surface")
plt.colorbar(im1, ax=axs[1])

plt.suptitle("GM 2025-04-01: True vs Predicted IV Surface")
plt.tight_layout()
plt.show()