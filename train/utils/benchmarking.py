from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_eval_metrics(y_true, y_pred):
    y_pred = torch.clamp(y_pred, min=0.001)
    y_true = y_true.detach().cpu().numpy().flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    r2 = r2_score(y_true, y_pred)
    
    return {"MAE": mae, "MSE": mse, "MAPE": mape, "R2": r2}

def plot_iv_surface(strikes, maturities, ivs, title="IV Surface", figsize=(8,6)):
    """
    Plots a 3D implied volatility surface.

    strikes, maturities, ivs should be 1D arrays of equal length.
    """
    assert len(strikes) == len(maturities) == len(ivs)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(strikes, maturities, ivs, cmap='viridis', edgecolor='none')

    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()