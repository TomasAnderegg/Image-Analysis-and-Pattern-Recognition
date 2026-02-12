import torch
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

from src.model import initialize_model
from src.data_related import TEST_IMAGE_DIR, get_transform, get_dataloader, NUM_CLASSES

# List of class names (update if needed)
CHOCOLATE_NAMES = [
    "Jelly White",
    "Jelly Milk",
    "Jelly Black",
    "Amandina",
    "Crème brulée",
    "Triangolo",
    "Tentation noir",
    "Comtesse",
    "Noblesse",
    "Noir authentique",
    "Passion au lait",
    "Arabia",
    "Stracciatella",
]


def run_inference_and_generate_csv(
    model_path, output_csv="submission.csv", device=None
):
    """
    Loads a trained model, runs inference on the test set, and writes predictions to a CSV.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = initialize_model(num_classes=NUM_CLASSES, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare test DataLoader
    test_loader = get_dataloader(
        csv_file=None,
        image_dir=TEST_IMAGE_DIR,
        transform=get_transform(train=False),
        batch_size=32,
        shuffle=False,
        test_mode=True,
    )

    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Test Inference"):
            images = images.to(device)
            outputs = model(images)
            # For regression: round and clip to int (no argmax, no softmax)
            counts = np.round(np.clip(outputs.cpu().numpy(), 0, None)).astype(int)
            for i, fname in enumerate(filenames):
                # Extract image ID (remove extension, remove leading 'L' if present)
                image_id = os.path.splitext(os.path.basename(fname))[0]
                if image_id.startswith("L"):
                    image_id = image_id[1:]
                image_ids.append(int(image_id))
                predictions.append(counts[i].tolist())

    # Sort by image ID to ensure correct order
    df = pd.DataFrame(predictions, columns=CHOCOLATE_NAMES)
    df.insert(0, "id", image_ids)
    df = df.sort_values("id")
    df.to_csv(output_csv, index=False)
    print(f"Submission CSV generated: {output_csv}")


if __name__ == "__main__":
    run_inference_and_generate_csv(
        model_path="best_model.pth", output_csv="submission.csv"
    )
