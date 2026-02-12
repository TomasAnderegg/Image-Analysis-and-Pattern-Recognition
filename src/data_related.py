import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset import ImageWithAttributesDataset

# ---- PATH SETUP ----
DATASET_DIR = r"C:\Users\Rayan\Desktop\Master\4_Image_analysis_and_pattern_recognition\Labs\EE451-IAPR\project\data"

AUGM_ROOT = os.path.join(DATASET_DIR, "augmented")
TRAIN_IMAGE_DIR = os.path.join(AUGM_ROOT, "training2")  # Use augmented training images
TRAIN_CSV = os.path.join(
    AUGM_ROOT, "augmented_training2.csv"
)  # Use augmented training CSV

# TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, "training2")
# TRAIN_CSV = os.path.join(DATASET_DIR, "training2.csv")

VAL_IMAGE_DIR = os.path.join(DATASET_DIR, "validation2")
VAL_CSV = os.path.join(DATASET_DIR, "validation2.csv")

TEST_IMAGE_DIR = os.path.join(DATASET_DIR, "test_og")

# ---- TRANSFORMS ----

NUM_CLASSES = 13  # Number of classes in the dataset
IMAGE_SIZE = 1024  # Image size for resizing ( intially 6000x4000)
NORM_MEAN = [0.6826428771018982, 0.6586042046546936, 0.6518846154212952]
NORM_STD = [0.1554882526397705, 0.15742455422878265, 0.17907005548477173]


def get_transform(train=True):
    """
    Returns the appropriate torchvision transforms for training or validation/test.

    Args:
        train (bool): Whether to return training transforms.

    Returns:
        torchvision.transforms.Compose: The composed transforms.
    """
    if train:
        return transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(ANGLE_RANGE),
                # transforms.Resize(IMAGE_SIZE), #shorter side is IMAGE_SIZE, aspect ratio IS preserved.
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ]
        )


# ---- DATALOADER CREATION ----
def get_dataloader(
    csv_file,
    image_dir,
    transform,
    batch_size=16,
    shuffle=True,
    test_mode=False,
    num_workers=4,
):  # num_workers= num of processor cores
    """
    Creates a DataLoader for the given dataset.

    Args:
        csv_file (str): Path to the CSV file with annotations (None for test).
        image_dir (str): Directory with all the images.
        transform (callable): Transformations to apply to the images.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        test_mode (bool): If True, disables loading targets.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: The PyTorch DataLoader.
    """
    dataset = ImageWithAttributesDataset(
        csv_file=csv_file, image_dir=image_dir, transform=transform, test_mode=test_mode
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
