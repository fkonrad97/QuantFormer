import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.synthetic.learnable.learnable_synthetic_iv_surface_dataset import LearnableIVSurfaceGenerator, IVSurfaceSampler, SyntheticIVDataset
import pandas as pd
import numpy as np
import os
import seaborn as sns
import torch.nn.functional as F

def load_all_iv_surfaces(folder_root_path, instruments=None):
    all_records = []
    for root, _, files in os.walk(folder_root_path):
        for file in files:
            if file.endswith('.csv') and "_iv_surface" in file:
                try:
                    parts = file.split("_")
                    if len(parts) < 3:
                        print(f"Invalid filename format: {file}")
                        continue
                    ticker = parts[0]
                    date_str = parts[1]
                    if instruments is not None and ticker not in instruments:
                        continue
                    filepath = os.path.join(root, file)
                    df = pd.read_csv(filepath)
                    df['ticker'] = ticker
                    df['date'] = date_str
                    all_records.append(df[['strike', 'maturity', 'iv', 'ticker', 'date']])
                except Exception as e:
                    print(f"Skipping {file}: {e}")
    if not all_records:
        print("No IV surface files found. Check folder path or instrument filters.")
        return pd.DataFrame(columns=['strike', 'maturity', 'iv', 'ticker', 'date'])
    print(f"Loaded {len(all_records)} files into dataframe")
    return pd.concat(all_records, ignore_index=True)

def train_learnable_generator(folder_path, tickers, num_assets=2, num_surfaces=500, num_points=300,
                              batch_size=8, lr=1e-2, epochs=200, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    df = load_all_iv_surfaces(folder_path, instruments=tickers)
    strike_cols = [col for col in df.columns if col.startswith("strike")]
    if not strike_cols:
        strike_vals = df["strike"].values
    else:
        strike_vals = df[strike_cols].values.flatten()

    maturity_vals = df["maturity"].values
    iv_vals = df["iv"].values

    sampler = IVSurfaceSampler(strike_vals, maturity_vals)
    generator = LearnableIVSurfaceGenerator(num_assets=num_assets, device=device).to(device)
    dataset = SyntheticIVDataset(generator, sampler, num_surfaces=num_surfaces, num_points=num_points)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(generator.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()

    target_iv_tensor = torch.tensor(iv_vals, dtype=torch.float32, device=device)
    target_iv_tensor = target_iv_tensor[torch.randperm(target_iv_tensor.size(0))[:1000]]

    print(f"Target IV stats: mean={target_iv_tensor.mean():.4f}, std={target_iv_tensor.std():.4f}")

    def mmd_loss(p_samples, q_samples, sigma=1.0):
        def gaussian_kernel(x, y):
            x = x.unsqueeze(1)
            y = y.unsqueeze(0)
            return torch.exp(-((x - y) ** 2) / (2 * sigma ** 2))
        Kxx = gaussian_kernel(p_samples, p_samples).mean()
        Kyy = gaussian_kernel(q_samples, q_samples).mean()
        Kxy = gaussian_kernel(p_samples, q_samples).mean()
        return Kxx + Kyy - 2 * Kxy

    train_losses = []
    for epoch in range(1, epochs + 1):
        generator.train()
        running_loss = 0.0
        for coords_batch, ivs_batch in dataloader:
            coords_batch = coords_batch.to(device)
            ivs_batch = ivs_batch.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                K_min = coords_batch[:, :, :-1].min(dim=1, keepdim=True).values
                K_max = coords_batch[:, :, :-1].max(dim=1, keepdim=True).values
                K_atm_batch = (K_min + K_max) / 2

            preds = generator(coords_batch, K_atm=K_atm_batch)

            mse_loss = mse_criterion(preds, ivs_batch)
            mmd = mmd_loss(preds.view(-1), target_iv_tensor.view(-1))
            loss = mse_loss + 0.1 * mmd

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        train_losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Total Loss = {avg_loss:.6f}")

    plt.plot(train_losses, label="MSE + MMD Loss")
    plt.title("Training Loss for Learnable IV Generator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluate over multiple surfaces
    generator.eval()
    generated_all = []
    with torch.no_grad():
        for _ in range(10):
            eval_coords = sampler.sample(1000, num_assets).unsqueeze(0).to(device)
            K_min = eval_coords[:, :, :-1].min(dim=1, keepdim=True).values
            K_max = eval_coords[:, :, :-1].max(dim=1, keepdim=True).values
            K_atm_batch = (K_min + K_max) / 2
            gen_ivs = generator(eval_coords, K_atm=K_atm_batch).squeeze(0).cpu().numpy()
            generated_all.append(gen_ivs)

    generated_all = np.concatenate(generated_all)

    plt.figure(figsize=(12, 5))
    sns.kdeplot(target_iv_tensor.cpu().numpy(), label="Real IVs", linewidth=2)
    sns.kdeplot(generated_all, label="Generated IVs (Avg of 10 Surfaces)", linewidth=2)
    plt.title("KDE of Real vs Generated IVs (Aggregated)")
    plt.xlabel("Implied Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return generator

if __name__ == "__main__":
    trained_generator = train_learnable_generator(
        folder_path="data/real_world/iv_data",
        tickers=["MS", "JPM"],
        num_assets=2,
        num_surfaces=200,
        num_points=300,
        batch_size=16,
        lr=1e-2,
        epochs=1000,
        device="cuda"
    )