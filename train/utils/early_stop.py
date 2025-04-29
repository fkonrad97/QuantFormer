import torch

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5, save_path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change to qualify as improvement.
            save_path (str): Where to save the best model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            # Save best model
            torch.save(model.state_dict(), self.save_path)
            print(f"Validation loss improved; saving model to {self.save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True