import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessor.general_iv_surface_input_processor import GeneralInputProcessor
from model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from data.real_world.real_iv_surface_dataset import RealIVSurfaceDataset
from data.real_world.iv_surface_data_processor import load_all_iv_surfaces

from train.utils.custom_scheduler import get_linear_warmup_scheduler
from train.utils.early_stop import EarlyStopping

import matplotlib.pyplot as plt
import os

def train_real_world_model(
    real_data_folder="data/real_world/iv_data",
    embed_dim=64,
    num_epochs=300,
    batch_size=128,
    learning_rate=1e-3,
    patience=30,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    if not os.path.exists(real_data_folder):
        raise ValueError(f"Folder {real_data_folder} does not exist!")

    df = load_all_iv_surfaces(real_data_folder)
    if df.empty:
        raise ValueError("Loaded real-world IV surface dataframe is empty!")

    dataset = RealIVSurfaceDataset(df, normalize=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Build Model ===
    input_processor = GeneralInputProcessor(
        input_dim=2,  # Strike + Maturity
        embed_dim=embed_dim,
        use_mlp=True
    )

    model = GeneralIVSurfaceTransformerModel(
        input_processor=input_processor,
        embed_dim=embed_dim,
        num_heads=4,
        ff_hidden_dim=embed_dim * 2,
        num_layers=4,
        head_hidden_dim=embed_dim,
        dropout=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    early_stopper = EarlyStopping(patience=patience, min_delta=1e-5)

    train_losses = []
    val_losses = []

    # === Training Loop ===
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for features_batch, targets_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            # Reshape: (batch_size, D) -> (1, batch_size, D)
            features_batch = features_batch.unsqueeze(0)
            targets_batch = targets_batch.unsqueeze(0)

            optimizer.zero_grad()

            preds = model(features_batch)
            loss = criterion(preds, targets_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation Pass ===
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

        # Early Stopping
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print("Real-world Training Complete!")

    # === Plot Training and Validation Loss ===
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Real-World IV Surface Training Losses")
    plt.legend()
    plt.grid()
    plt.show()

    return early_stopper.best_model

if __name__ == "__main__":
    print("Training on real-world IV surfaces...")
    best_model = train_real_world_model()
