import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

from preprocessor.general_iv_surface_input_processor import GeneralInputProcessor
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor
from model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from data.synthetic.general_synthetic_iv_surface_dataset import GeneralSyntheticAssetIVSurfaceDataset, general_collate_fn
from train.utils.custom_scheduler import get_linear_warmup_scheduler
from train.utils.early_stop import EarlyStopping
from train.utils.benchmarking import compute_eval_metrics
from data.real_world.real_iv_multiasset_dataset import RealMultiAssetIVSurfaceDataset
from data.real_world.iv_surface_data_processor import build_iv_dataset

import matplotlib.pyplot as plt

def train_on_synthetic_multi_asset(save_path="synthetic_multiasset_pretrained.pth", 
                                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    print("Pretraining on multi-asset synthetic surfaces...")

    # Build multi-asset synthetic dataset
    dataset = GeneralSyntheticAssetIVSurfaceDataset(
        min_assets=1,           # now using multiple assets
        max_assets=3,
        num_strikes_per_asset=6,
        num_maturities=6,
        k_min=90,
        k_max=140,
        t_min=0.05,
        t_max=2.5,
        base_vol=np.random.uniform(0.30, 0.36),
        skew_strength=-0.0008,
        term_slope=0.002,
        n_surfaces=300,
        device='cuda',
        randomize_skew=True,
        randomize_term=True,
        add_noise=True,
        noise_level=0.004,
        normalize=True  # normalize synthetic data to standardize targets
    )

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # The collate function will pad features and compute asset_dims for each sample.
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=general_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=general_collate_fn)

    # For multi-asset data, using the MaskedGeneralInputProcessor remains appropriate.
    input_processor = MaskedGeneralInputProcessor(
        input_dim=dataset[0][0].shape[-1],
        embed_dim=64,
        use_mlp=True
    )

    model = GeneralIVSurfaceTransformerModel(
        input_processor=input_processor,
        embed_dim=64,
        num_heads=2,
        ff_hidden_dim=128,
        num_layers=4,
        head_hidden_dim=64,
        dropout=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    total_steps = 300 * len(train_loader)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
    early_stopper = EarlyStopping(patience=30, min_delta=1e-5, save_path=save_path)

    train_losses, val_losses = [], []
    # Optionally, track additional metrics on the validation set
    val_metrics_list = []

    for epoch in range(1, 301):
        model.train()
        running_loss = 0.0

        for features_batch, targets_batch, asset_dims_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()

            preds = model(features_batch, asset_dims_batch)
            loss = criterion(preds, targets_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation Pass ===
        model.eval()
        running_val_loss = 0.0
        all_metrics = []
        with torch.no_grad():
            for features_batch, targets_batch, asset_dims_batch in val_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                preds = model(features_batch, asset_dims_batch)

                # For evaluation, convert normalized IVs back to real scale
                preds_real = dataset.inverse_transform_iv(preds)
                targets_real = dataset.inverse_transform_iv(targets_batch)
                
                val_loss = criterion(preds_real, targets_real)
                running_val_loss += val_loss.item()

                # Compute additional metrics from real values
                metrics = compute_eval_metrics(targets_real, preds_real)
                all_metrics.append(metrics)

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # Optionally compute average metrics over validation set
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]
        }
        val_metrics_list.append(avg_metrics)

        if epoch % 10 == 1:
            print(f"Epoch {epoch:03d}: Train Loss (norm) = {avg_train_loss:.8f}, Val Loss (real) = {avg_val_loss:.8f}, "
                  f"MAE = {avg_metrics['MAE']:.4f}, MAPE = {avg_metrics['MAPE']:.2f}%, R² = {avg_metrics['R2']:.4f}")
            print(f"[Epoch {epoch:03d}] Pred Real IV → min={preds_real.min():.4f}, max={preds_real.max():.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Multi-asset synthetic pretraining finished. Model saved to {save_path}")

    # === Dual Plot of Training Metrics (if needed) ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(train_losses, label="Train Loss (normalized)")
    axs[0].set_title("Training Loss (Normalized)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE (Normalized)")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(val_losses, label="Val Loss (real IV units)")
    axs[1].set_title("Validation Loss (Real IV)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MSE (Real IV)")
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    plt.show()

    return save_path

def finetune_on_real_world_multi_asset(pretrained_model_path,
                                       folder_path,
                                       tickers,
                                       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                       save_path="finetuned_multiasset_model.pth"):
    """
    Fine-tunes a pretrained multi-asset model on real-world multi-asset IV surface data.

    Args:
        pretrained_model_path (str): Path to the pretrained model weights.
        folder_path (str): Path to real-world data folder (e.g., "data/real_world/iv_data").
        tickers (list[str]): List of tickers to use for multi-asset surface construction.
        device (torch.device): Device to use.
        save_path (str): Where to save the final fine-tuned model.
    """
    print(f"\nFine-tuning on real-world multi-asset IV data: {tickers}")

    # === Step 1: Build and load dataset === #
    df = build_iv_dataset(folder_path, tickers, method="mean")
    assert not df.empty, "The multi-asset dataset is empty!"

    dataset = RealMultiAssetIVSurfaceDataset(df, normalize=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # === Step 2: Build model === #
    input_processor = MaskedGeneralInputProcessor(
        input_dim=dataset[0][0].shape[-1],
        embed_dim=64,
        use_mlp=True
    )

    model = GeneralIVSurfaceTransformerModel(
        input_processor=input_processor,
        embed_dim=64,
        num_heads=4,
        ff_hidden_dim=128,
        num_layers=4,
        head_hidden_dim=64,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load(pretrained_model_path))
    print(f"Loaded pretrained weights from {pretrained_model_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    total_steps = 100 * len(train_loader)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
    early_stopper = EarlyStopping(patience=20, min_delta=1e-5, save_path=save_path)

    train_losses, val_losses = [], []
    val_metrics_list = []

    for epoch in range(1, 101):
        model.train()
        running_loss = 0.0
        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            features_batch = features_batch.unsqueeze(0)
            targets_batch = targets_batch.unsqueeze(0)
            asset_dims = [features_batch.shape[-1] - 1]  # exclude maturity

            optimizer.zero_grad()
            preds = model(features_batch, asset_dims=asset_dims)
            loss = criterion(preds, targets_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation
        model.eval()
        running_val_loss = 0.0
        all_metrics = []
        with torch.no_grad():
            for features_batch, targets_batch in val_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                features_batch = features_batch.unsqueeze(0)
                targets_batch = targets_batch.unsqueeze(0)
                asset_dims = [features_batch.shape[-1] - 1]

                preds = model(features_batch, asset_dims=asset_dims)

                preds_real = dataset.inverse_transform_iv(preds)
                targets_real = dataset.inverse_transform_iv(targets_batch)

                val_loss = criterion(preds_real, targets_real)
                running_val_loss += val_loss.item()

                metrics = compute_eval_metrics(targets_real, preds_real)
                all_metrics.append(metrics)

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        val_metrics_list.append(avg_metrics)

        if epoch % 10 == 1:
            print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.8f}, Val Loss = {avg_val_loss:.8f}, "
                  f"MAE = {avg_metrics['MAE']:.4f}, MAPE = {avg_metrics['MAPE']:.2f}%, R² = {avg_metrics['R2']:.4f}")
            print(f"[Epoch {epoch:03d}] Pred Real IV → min={preds_real.min():.4f}, max={preds_real.max():.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Fine-tuning complete.")
    print(f"Final model saved to {save_path}")

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(train_losses, label="Train Loss (norm)")
    axs[0].set_title("Training Loss")
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(val_losses, label="Validation Loss (real IV)")
    axs[1].set_title("Validation Loss")
    axs[1].grid()
    axs[1].legend()

    plt.suptitle(f"Fine-tuning on Real IV Data ({'_'.join(tickers)})")
    plt.tight_layout()
    plt.show()

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Run training for multi-asset synthetic data
    pretrained_model_path = train_on_synthetic_multi_asset(save_path="synthetic_multiasset_pretrained.pth", device=device)
    finetune_on_real_world_multi_asset(
        pretrained_model_path=pretrained_model_path,
        folder_path="data/real_world/iv_data",
        tickers=["MS", "JPM", "GM"],
        save_path="finetuned_multiasset_model.pth"
    )

if __name__ == "__main__":
    main()