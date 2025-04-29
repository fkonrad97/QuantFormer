import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from preprocessor.general_iv_surface_input_processor import GeneralInputProcessor
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor
from model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from data.synthetic.general_synthetic_iv_surface_dataset import GeneralSyntheticAssetIVSurfaceDataset, general_collate_fn
from data.real_world.real_iv_surface_dataset import RealIVSurfaceDataset
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces

from train.utils.custom_scheduler import get_linear_warmup_scheduler
from train.utils.early_stop import EarlyStopping

import matplotlib.pyplot as plt

def train_on_synthetic(save_path="synthetic_pretrained.pth", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    print("Pretraining on synthetic surfaces...")

    dataset = GeneralSyntheticAssetIVSurfaceDataset(
        min_assets=1,
        max_assets=1,
        num_strikes_per_asset=10,
        num_maturities=10,
        n_surfaces=500,
        device=device,
        randomize_skew=True,
        randomize_term=True,
        add_noise=True,
        noise_level=0.01
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
    early_stopper = EarlyStopping(patience=30, min_delta=1e-5)

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

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features_batch, targets_batch, asset_dims_batch in val_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                preds = model(features_batch, asset_dims_batch)
                val_loss = criterion(preds, targets_batch)

                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 100 == 1:
            print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.8f}, Val Loss = {avg_val_loss:.8f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Save best model
    torch.save(early_stopper.best_model.state_dict(), save_path)
    print(f"Synthetic pretraining finished. Model saved to {save_path}")

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Synthetic Pretraining Losses")
    plt.legend()
    plt.grid()
    plt.show()

    return save_path

def finetune_on_real_world(pretrained_model_path, real_data_folder="data/real_iv_surfaces", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    print("\nFinetuning on real-world IV data...")

    if not os.path.exists(real_data_folder):
        raise ValueError(f"Folder {real_data_folder} does not exist!")

    df = load_all_iv_surfaces(real_data_folder)
    if df.empty:
        raise ValueError("Real IV surface dataframe is empty!")

    dataset = RealIVSurfaceDataset(df, normalize=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    input_processor = GeneralInputProcessor(
        input_dim=2,  # Strike + Maturity
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

    # Load pretrained weights
    model.load_state_dict(torch.load(pretrained_model_path))
    print(f"Loaded pretrained weights from {pretrained_model_path}")

    optimizer = optim.Adam(model.parameters(), lr=5e-5)  # smaller LR for finetuning
    criterion = nn.MSELoss()

    total_steps = 100 * len(train_loader)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps)
    early_stopper = EarlyStopping(patience=20, min_delta=1e-5)

    train_losses, val_losses = [], []

    for epoch in range(1, 101):
        model.train()
        running_loss = 0.0

        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            features_batch = features_batch.unsqueeze(0)
            targets_batch = targets_batch.unsqueeze(0)

            optimizer.zero_grad()

            preds = model(features_batch)
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

                preds = model(features_batch)
                val_loss = criterion(preds, targets_batch)

                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 100 == 1:
            print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.8f}, Val Loss = {avg_val_loss:.8f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Finetuning complete.")

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Real-World Finetuning Losses")
    plt.legend()
    plt.grid()
    plt.show()

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_model_path = train_on_synthetic(device=device)

    best_model = finetune_on_real_world(pretrained_model_path, device=device)

if __name__ == "__main__":
    main()