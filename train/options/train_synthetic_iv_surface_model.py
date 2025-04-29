import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocessor.general_iv_surface_input_processor import GeneralInputProcessor
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor
from model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from data.synthetic.general_synthetic_iv_surface_dataset import GeneralSyntheticAssetIVSurfaceDataset, general_collate_fn

from train.utils.custom_scheduler import get_linear_warmup_scheduler
from train.utils.early_stop import EarlyStopping

import matplotlib.pyplot as plt

def train_synthetic_model(
    use_masked=True,
    embed_dim=64,
    num_epochs=300,
    batch_size=2,
    learning_rate=1e-3,
    patience=30,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    # === 1. Prepare Dataset ===
    dataset = GeneralSyntheticAssetIVSurfaceDataset(
        min_assets=1,
        max_assets=1,
        num_strikes_per_asset=10,
        num_maturities=10,
        n_surfaces=200,
        device=device,
        randomize_skew=True,
        randomize_term=True,
        add_noise=True,
        noise_level=0.01
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=general_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=general_collate_fn)

    # === 2. Build Model ===
    input_processor = MaskedGeneralInputProcessor(
        input_dim=dataset[0][0].shape[-1],
        embed_dim=embed_dim,
        use_mlp=True
    ) if use_masked else GeneralInputProcessor(
        input_dim=dataset[0][0].shape[-1],
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

    # Scheduler and Early Stopping
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    early_stopper = EarlyStopping(patience=patience, min_delta=1e-5)

    train_losses = []
    val_losses = []

    # === 3. Training Loop ===
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        for features_batch, targets_batch, asset_dims_batch in train_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()

            preds = model(features_batch, asset_dims_batch if use_masked else None)
            loss = criterion(preds, targets_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === 4. Validation Pass ===
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features_batch, targets_batch, asset_dims_batch in val_loader:
                features_batch = features_batch.to(device)
                targets_batch = targets_batch.to(device)

                preds = model(features_batch, asset_dims_batch if use_masked else None)
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

    print("Training Complete!")

    # === 5. Plot Training and Validation Loss ===
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Synthetic IV Surface Training Losses")
    plt.legend()
    plt.grid()
    plt.show()

    return early_stopper.best_model

if __name__ == "__main__":
    print("Training on synthetic surfaces (Masked Processor)...")
    best_model = train_synthetic_model(use_masked=True)