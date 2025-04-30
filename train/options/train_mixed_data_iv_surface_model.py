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
from data.real_world.real_iv_surface_dataset import RealIVSurfaceDataset
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces

from train.utils.custom_scheduler import get_linear_warmup_scheduler
from train.utils.early_stop import EarlyStopping
from train.utils.benchmarking import compute_eval_metrics, plot_iv_surface

import matplotlib.pyplot as plt

def train_on_synthetic(save_path="synthetic_pretrained.pth", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    print("Pretraining on synthetic surfaces...")

    # === Build dataset === #
    dataset = GeneralSyntheticAssetIVSurfaceDataset(
        min_assets=1,
        max_assets=1,
        num_strikes_per_asset=10,
        num_maturities=10,
        k_min=90,
        k_max=140,
        t_min=0.05,
        t_max=2.5,
        base_vol=np.random.uniform(0.30, 0.36),
        skew_strength=-0.0008,
        term_slope=0.002,
        n_surfaces=200,
        device='cuda',
        randomize_skew=True,
        randomize_term=True,
        add_noise=True,
        noise_level=0.004,
        normalize=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=general_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=general_collate_fn)

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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    total_steps = 300 * len(train_loader)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
    early_stopper = EarlyStopping(patience=30, min_delta=1e-5, save_path=save_path)

    train_losses, val_losses = [], []

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

        # === Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features_batch, targets_batch, asset_dims_batch in val_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                preds = model(features_batch, asset_dims_batch)

                # === Convert normalized → real IVs
                preds_real = dataset.inverse_transform_iv(preds)
                targets_real = dataset.inverse_transform_iv(targets_batch)

                val_loss = criterion(preds_real, targets_real)
                metrics = compute_eval_metrics(targets_real, preds_real)

                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 10 == 1:
            print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.8f}, "
                f"Val Loss = {avg_val_loss:.8f}, "
                f"MAE = {metrics['MAE']:.4f}, "
                f"MAPE = {metrics['MAPE']:.2f}%, "
                f"R² = {metrics['R2']:.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Synthetic pretraining finished. Model saved to {save_path}")

    # === Dual loss tracking
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Normalized MSE (training)
    axs[0].plot(train_losses, label="Training Loss (normalized IV space)", color="blue")
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE (normalized)")
    axs[0].grid()
    axs[0].legend()

    # Plot 2: Real-space MSE (validation)
    axs[1].plot(val_losses, label="Validation Loss (real IV space)", color="green")
    axs[1].set_title("Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MSE (real IV units)")
    axs[1].grid()
    axs[1].legend()

    plt.suptitle("Finetuning Loss Curves (Synthetic)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    return save_path

def finetune_on_real_world(pretrained_model_path, real_data_folder="data/real_world/iv_data", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), save_path="fine_tuned_model.pth"):
    print("\nFinetuning on real-world IV data...")

    if not os.path.exists(real_data_folder):
        raise ValueError(f"Folder {real_data_folder} does not exist!")

    df = load_all_iv_surfaces(real_data_folder, instruments=["MS", "JPM", "GM"])
    if df.empty:
        raise ValueError("Real IV surface dataframe is empty!")

    dataset = RealIVSurfaceDataset(df, normalize=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # === Use SAME processor as pretraining ===
    input_processor = MaskedGeneralInputProcessor(
        input_dim=2,  # strike + maturity
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

    # === Load pretrained weights ===
    model.load_state_dict(torch.load(pretrained_model_path))
    print(f"Loaded pretrained weights from {pretrained_model_path}")

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    total_steps = 100 * len(train_loader)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
    early_stopper = EarlyStopping(patience=20, min_delta=1e-5, save_path=save_path)

    train_losses, val_losses = [], []

    for epoch in range(1, 101):
        model.train()
        running_loss = 0.0

        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)             # shape: (B, 2)
            targets_batch = targets_batch.to(device)               # shape: (B,)

            features_batch = features_batch.unsqueeze(0)           # → (1, B, 2)
            targets_batch = targets_batch.unsqueeze(0)             # → (1, B)
            asset_dims = [1]                                       # 1 asset + maturity

            optimizer.zero_grad()
            preds = model(features_batch, asset_dims=asset_dims)
            loss = criterion(preds, targets_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features_batch, targets_batch in val_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                features_batch = features_batch.unsqueeze(0)
                targets_batch = targets_batch.unsqueeze(0)
                asset_dims = [1]

                preds = model(features_batch, asset_dims=asset_dims)

                # Invert predicted and target IVs to real scale for evaluation
                preds_real = dataset.inverse_transform_iv(preds)
                targets_real = dataset.inverse_transform_iv(targets_batch)
                
                val_loss = criterion(preds_real, targets_real)
                metrics = compute_eval_metrics(targets_real, preds_real)

                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 10 == 1:
            print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.8f}, "
                  f"Val Loss = {avg_val_loss:.8f}, "
                  f"MAE = {metrics['MAE']:.4f}, "
                  f"MAPE = {metrics['MAPE']:.2f}%, "
                  f"R² = {metrics['R2']:.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Finetuning complete.")

    # Save final model
    print(f"Finetuned model saved to {save_path}")

    # === Dual loss tracking 
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Normalized MSE (training)
    axs[0].plot(train_losses, label="Training Loss (normalized IV space)", color="blue")
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE (normalized)")
    axs[0].grid()
    axs[0].legend()

    # Plot 2: Real-space MSE (validation)
    axs[1].plot(val_losses, label="Validation Loss (real IV space)", color="green")
    axs[1].set_title("Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MSE (real IV units)")
    axs[1].grid()
    axs[1].legend()

    plt.suptitle("Finetuning Loss Curves (Real-World)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model_path = train_on_synthetic(device=device)

    finetune_on_real_world(pretrained_model_path, device=device, save_path="finetuned_model.pth")

if __name__ == "__main__":
    main()