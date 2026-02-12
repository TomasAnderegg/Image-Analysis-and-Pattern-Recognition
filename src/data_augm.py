import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# === Configuration ===
DATASET_DIR = r"C:\Users\Rayan\Desktop\Master\4_Image_analysis_and_pattern_recognition\Labs\EE451-IAPR\project\data"
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "training2")
TRAIN_CSV = os.path.join(DATASET_DIR, "training2.csv")

AUGM_ROOT = os.path.join(DATASET_DIR, "augmented")
TRAIN_AUGM_DIR = os.path.join(AUGM_ROOT, "training2")

os.makedirs(TRAIN_AUGM_DIR, exist_ok=True)


BRIGHTNESS = 0.25 # <0.25 for good results
CONTRAST = 0.5 # <0.5 for good results
SATURATION = 0.2 # <0.2 for good results
ANGLE_RANGE = 7  # degrees
IMAGE_SIZE = 1024  # Image size for resizing (initially 6000x4000)
AUGMENTED_IMAGES = 15  # Number of augmented images per original image

# ---- Compute mean and std for your dataset ----
def compute_mean_std(image_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    means = []
    stds = []
    for img_path in tqdm(image_files, desc="Computing mean/std"):
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)
        means.append(img.mean(dim=(1,2)))
        stds.append(img.std(dim=(1,2)))
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    print("Dataset mean:", mean.tolist())
    print("Dataset std:", std.tolist())
    return mean.tolist(), std.tolist()

def augment_dataset(source_dir, csv_path, target_dir, num_augmented_per_image, output_csv):
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str)  
    id_to_row = {str(int(row["id"])): row for _, row in df.iterrows()}

    image_paths = []
    valid_ids = []

    for f in os.listdir(source_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = os.path.splitext(f)[0]
            numeric_id = ''.join(filter(str.isdigit, filename))
            if numeric_id in id_to_row:
                image_paths.append(os.path.join(source_dir, f))
                valid_ids.append(numeric_id)

    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(ANGLE_RANGE),
        transforms.ColorJitter(
            brightness=BRIGHTNESS,
            contrast=CONTRAST,
            saturation=SATURATION,
        ),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])

    resize_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])

    augmented_data = []

    # 1. Copy original images (with resizing and normalization) and add their metadata
    for img_path, numeric_id in zip(image_paths, valid_ids):
        orig_filename = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        resized_img = resize_transform(img)
        save_path = os.path.join(target_dir, orig_filename)
        resized_img.save(save_path)
        row_data = id_to_row[numeric_id]
        row_copy = row_data.copy()
        row_copy["id"] = numeric_id
        augmented_data.append(row_copy)

    # 2. Generate augmented images and metadata with progress bar
    for img_path, numeric_id in tqdm(list(zip(image_paths, valid_ids)), desc="Augmenting images"):
        row_data = id_to_row[numeric_id]
        for aug_idx in range(num_augmented_per_image):
            img = Image.open(img_path).convert("RGB")
            aug_img = augment_transform(img)
            new_id = f"{numeric_id}_2{aug_idx}"
            new_filename = f"L{new_id}.jpg"
            save_path = os.path.join(target_dir, new_filename)
            aug_img.save(save_path)
            row_copy = row_data.copy()
            row_copy["id"] = new_id
            augmented_data.append(row_copy)

    df_augmented = pd.DataFrame(augmented_data)
    df_augmented.to_csv(output_csv, index=False)
    print(f"{len(image_paths)} original + {len(image_paths) * num_augmented_per_image} augmented images generated in {target_dir}")
    print(f"CSV saved at: {output_csv}")

if __name__ == "__main__":
    # Uncomment and run once to get your dataset's mean and std, then hardcode them in r_data.py
    compute_mean_std(TRAIN_IMAGE_DIR)

    # Augment training set: AUGMENTED_IMAGES augmentations per image (original + AUGMENTED_IMAGES)
    augment_dataset(
        source_dir=TRAIN_IMAGE_DIR,
        csv_path=TRAIN_CSV,
        target_dir=TRAIN_AUGM_DIR,
        num_augmented_per_image=AUGMENTED_IMAGES,  # AUGMENTED_IMAGES new per original
        output_csv=os.path.join(AUGM_ROOT, "augmented_training.csv")
    )
    # Do NOT augment validation set!