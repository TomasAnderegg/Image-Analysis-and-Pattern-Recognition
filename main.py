import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.optim as optim
import torch.nn as nn
import numpy as np

from src.data_related import (
    TRAIN_IMAGE_DIR,
    VAL_IMAGE_DIR,
    TRAIN_CSV,
    VAL_CSV,
    NUM_CLASSES,
    get_transform,
    get_dataloader,
)
from src.model import initialize_model
from src.training import train_model, train_one_epoch, validate
from src.test_dataset import run_inference_and_generate_csv
from src.utils.metrics import plot_multilabel_confusion_matrices,plot_metrics
from src.test_dataset import CHOCOLATE_NAMES

# Training with early stopping

import torch

N_EPOCHS = 1000  # Number of epochs for training
LEARNING_RATE = 1e-3  # Learning rate for optimizer
WEIGHT_DECAY = 1e-4  # Weight decay for optimizer
BATCH_SIZE = 4 # Batch size for training and validation
EARLY_STOPPING_PATIENCE = 20  # Number of epochs to wait for improvement
MIN_EPOCHS = 50  # Minimum epochs before early stopping can trigger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(torch.cuda.is_available())
    if device.type == "cuda":
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

    # Data loaders for train and validation
    train_loader = get_dataloader(
        csv_file=TRAIN_CSV,
        image_dir=TRAIN_IMAGE_DIR,
        transform=get_transform(train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        test_mode=False,
    )
    val_loader = get_dataloader(
        csv_file=VAL_CSV,
        image_dir=VAL_IMAGE_DIR,
        transform=get_transform(train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        test_mode=False,
    )

    # Model initialization
    model = initialize_model(num_classes=NUM_CLASSES, device=device)
    print("Model initialized.")

    # Display model summary and parameter count
    print(model)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

    # Train the model using the modular train_model function
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_epochs=MIN_EPOCHS,
        save_path="best_model.pth",
        verbose=True,
    )

    

    # Generate predictions for submission using the best model
    run_inference_and_generate_csv(
        model_path="best_model.pth", output_csv="submission.csv", device=device
    )

    # --- Plot and save F1 curves ---
    plot_metrics(
        result["train_losses"],
        result["val_losses"],
        result["train_f1s"],
        result["val_f1s"],
    )

    # --- Generate and save confusion matrices for validation set ---
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = np.round(np.clip(outputs.cpu().numpy(), 0, None)).astype(int)
            true = np.round(np.clip(targets.cpu().numpy(), 0, None)).astype(int)
            all_preds.append(preds)
            all_targets.append(true)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    class_names = CHOCOLATE_NAMES
    plot_multilabel_confusion_matrices(all_targets, all_preds, class_names)

if __name__ == "__main__":
    main()

