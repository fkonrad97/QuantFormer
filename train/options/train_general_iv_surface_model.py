import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pinnLib.transformer.data.synthetic.general_iv_surface_dataset import GeneralAssetIVSurfaceDataset, general_collate_fn
from pinnLib.transformer.model.option_models.general_iv_surface_transformer_model import GeneralIVSurfaceTransformerModel
from preprocessor.general_iv_surface_input_processor import GeneralInputProcessor
from preprocessor.masked_general_iv_surface_input_processor import MaskedGeneralInputProcessor
from train.utils.early_stop import EarlyStopping
from train.utils.custom_scheduler import get_linear_warmup_scheduler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====== PLOTTING UTILS ======

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_iv_surface(features, volatilities, num_strikes, num_maturities):
    """
    Plots a 3D implied volatility surface.

    Args:
        features (torch.Tensor): (num_points, input_dim), where input_dim=2 (strike, maturity).
        volatilities (torch.Tensor): (num_points,), implied volatilities.
        num_strikes (int): Number of strike points.
        num_maturities (int): Number of maturity points.
    """

    # Extract strike and maturity from features
    strikes = features[:, 0].cpu().numpy()
    maturities = features[:, 1].cpu().numpy()
    vols = volatilities.cpu().numpy()

    # Reshape into 2D grid
    strikes = strikes.reshape((num_strikes, num_maturities))
    maturities = maturities.reshape((num_strikes, num_maturities))
    vols = vols.reshape((num_strikes, num_maturities))

    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(strikes, maturities, vols, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Synthetic IV Surface')
    plt.show()

def plot_multiple_surfaces(dataset, num_samples=4, num_strikes=10, num_maturities=10):
    """
    Plots multiple IV surfaces from the dataset in a single figure with subplots.

    Args:
        dataset: Instance of GeneralAssetIVSurfaceDataset.
        num_samples: Number of surfaces to plot.
        num_strikes: Number of strikes (grid size).
        num_maturities: Number of maturities (grid size).
    """
    indices = torch.randperm(len(dataset))[:num_samples]

    cols = 2  # You can change to 3, 4, etc.
    rows = (num_samples + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    for i, idx in enumerate(indices):
        features, vols = dataset[idx]
        strikes = features[:, 0].cpu().numpy()
        maturities = features[:, 1].cpu().numpy()
        vols = vols.cpu().numpy()

        # Reshape
        strikes = strikes.reshape((num_strikes, num_maturities))
        maturities = maturities.reshape((num_strikes, num_maturities))
        vols = vols.reshape((num_strikes, num_maturities))

        # Subplot
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.plot_surface(strikes, maturities, vols, cmap='viridis')

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('IV')
        ax.set_title(f"Surface {i+1}")

    plt.tight_layout()
    plt.show()

def plot_true_vs_predicted(model, dataset, num_samples=3, num_strikes=10, num_maturities=10, device='cpu', use_masked_processor=False):
    """
    Plots true and predicted IV surfaces side-by-side for a few samples.

    Args:
        model: Trained model.
        dataset: Validation dataset (or any dataset).
        num_samples: Number of samples to plot.
        num_strikes: Number of strike points.
        num_maturities: Number of maturity points.
        device: 'cuda' or 'cpu'.
        use_masked_processor: Whether to use asset_dims with input_processor.
    """
    model.eval()

    indices = torch.randperm(len(dataset))[:num_samples]
    fig = plt.figure(figsize=(10, 5 * num_samples))

    for i, idx in enumerate(indices):
        features, targets = dataset[idx]

        features = features.unsqueeze(0).to(device)  # Add batch dim
        targets = targets.unsqueeze(0).to(device)

        if use_masked_processor:
            asset_dims = [features.shape[-1] - 1]  # Calculate asset dim dynamically
            embedded = model.input_processor(features, asset_dims)
        else:
            embedded = model.input_processor(features)

        preds = model.head(model.encoder(embedded))
        preds = preds.detach().cpu().squeeze(0)
        targets = targets.detach().cpu().squeeze(0)

        # Reshape
        strikes = features.squeeze(0)[:, 0].cpu().numpy()
        maturities = features.squeeze(0)[:, 1].cpu().numpy()

        strikes = strikes.reshape((num_strikes, num_maturities))
        maturities = maturities.reshape((num_strikes, num_maturities))
        preds = preds.reshape((num_strikes, num_maturities))
        targets = targets.reshape((num_strikes, num_maturities))

        # Plot true surface
        ax1 = fig.add_subplot(num_samples, 2, 2*i+1, projection='3d')
        ax1.plot_surface(strikes, maturities, targets, cmap='viridis')
        ax1.set_title(f"True Surface {i+1}")
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity')
        ax1.set_zlabel('IV')

        # Plot predicted surface
        ax2 = fig.add_subplot(num_samples, 2, 2*i+2, projection='3d')
        ax2.plot_surface(strikes, maturities, preds, cmap='plasma')
        ax2.set_title(f"Predicted Surface {i+1}")
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Maturity')
        ax2.set_zlabel('IV')

    plt.tight_layout()
    plt.show()


# ====== CONFIGURATION ======
use_masked_processor = True  # or False
use_mlp = False
embed_dim = 64
batch_size = 1
num_epochs = 1000
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== DATASET ======
dataset = GeneralAssetIVSurfaceDataset(
    min_assets=1,               # Only 1 asset
    max_assets=1,               # Only 1 asset
    num_strikes_per_asset=10,    # Reasonable grid
    num_maturities=10,           # Reasonable term structure
    n_surfaces=100,              # Enough surfaces for training
    randomize_skew=True,         # Realistic smiles
    randomize_term=True,         # Realistic term structure slopes
    add_noise=True,              # Realistic market noise
    noise_level=0.005            # Small Gaussian noise (~0.5%)
)

plot_multiple_surfaces(dataset, num_samples=4, num_strikes=10, num_maturities=10)

# Train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=general_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=general_collate_fn)

# ====== MODEL SETUP ======

# Peek first batch
features_batch, _, _ = next(iter(train_loader))
max_input_dim = features_batch.shape[-1]

if use_masked_processor:
    input_processor = MaskedGeneralInputProcessor(
        input_dim=max_input_dim,
        embed_dim=embed_dim,
        use_mlp=use_mlp
    )
else:
    input_processor = GeneralInputProcessor(
        input_dim=max_input_dim,
        embed_dim=embed_dim,
        use_mlp=use_mlp
    )

# Instantiate model
model = GeneralIVSurfaceTransformerModel(
    input_processor=input_processor,
    embed_dim=embed_dim,
    num_heads=4,
    ff_hidden_dim=128,
    num_layers=4,
    head_hidden_dim=64,
    dropout=0.1,
)
model = model.to(device)

# ====== OPTIMIZER / LOSS / EARLY STOP ======
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
early_stopper = EarlyStopping(patience=30, min_delta=1e-5)

# ====== LR SCHEDULER ======
total_training_steps = num_epochs * len(train_loader)
warmup_steps = int(0.1 * total_training_steps)  # 10% warmup

scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_training_steps)

# ====== TRAINING LOOP ======

train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    model.train()
    running_train_loss = 0.0

    for features_batch, targets_batch, asset_dims_batch in train_loader:
        features_batch = features_batch.to(device)
        targets_batch = targets_batch.to(device)

        # Process input
        if use_masked_processor:
            embedded = model.input_processor(features_batch, asset_dims_batch)
        else:
            embedded = model.input_processor(features_batch)

        embedded = embedded.to(device)

        # Forward pass
        preds = model.head(model.encoder(embedded))  # preds: (batch_size, seq_len)

        loss = criterion(preds, targets_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ====== Validation ======
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for features_batch, targets_batch, asset_dims_batch in val_loader:
            features_batch = features_batch.to(device)
            targets_batch = targets_batch.to(device)

            if use_masked_processor:
                embedded = model.input_processor(features_batch, asset_dims_batch)
            else:
                embedded = model.input_processor(features_batch)

            embedded = embedded.to(device)

            preds = model.head(model.encoder(embedded))

            val_loss = criterion(preds, targets_batch)

            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Early stop check
    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

    if epoch % 100 == 0 or epoch == 1 or epoch == num_epochs:
        print(f"Epoch {epoch:04d}: Train Loss = {avg_train_loss:.10f}, Val Loss = {avg_val_loss:.10f}")

print("Training complete!")

plot_losses(train_losses, val_losses)
plot_true_vs_predicted(model, val_dataset, num_samples=3, num_strikes=10, num_maturities=10, device=device, use_masked_processor=True)