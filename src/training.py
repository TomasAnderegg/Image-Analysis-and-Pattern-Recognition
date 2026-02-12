import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.metrics import plot_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from src.utils.metrics import imagewise_f1_score


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        # Mixed precision context
        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        # Backward with scaler
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)

        outputs_np = np.round(np.clip(outputs.detach().cpu().numpy(), 0, None)).astype(
            int
        )
        targets_np = np.round(np.clip(targets.detach().cpu().numpy(), 0, None)).astype(
            int
        )
        all_preds.append(outputs_np)
        all_targets.append(targets_np)
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    epoch_f1 = imagewise_f1_score(all_targets, all_preds)
    return epoch_loss, epoch_f1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Val", leave=False):
            images, targets = images.to(device), targets.to(device)
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            outputs_np = np.round(np.clip(outputs.cpu().numpy(), 0, None)).astype(int)
            targets_np = np.round(np.clip(targets.cpu().numpy(), 0, None)).astype(int)
            all_preds.append(outputs_np)
            all_targets.append(targets_np)
    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    epoch_f1 = imagewise_f1_score(all_targets, all_preds)
    return epoch_loss, epoch_f1

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs=1000,
    lr=1e-3,
    weight_decay=1e-4,
    early_stopping_patience=20,
    min_epochs=50,
    criterion=None,
    save_path="best_model.pth",
    verbose=True,
):
    if criterion is None:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -float("inf")
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []
    scaler = GradScaler() if device.type == "cuda" else None
    for epoch in range(n_epochs):
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs}")
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler
        )
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        if verbose:
            print(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            if verbose:
                print("Best model saved.")
        elif epoch >= min_epochs - 1:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1}."
                    )
                break

    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_f1s": train_f1s,
        "val_f1s": val_f1s,
        "save_path": save_path,
    }
