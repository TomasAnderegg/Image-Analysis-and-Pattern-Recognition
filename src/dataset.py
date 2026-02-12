import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class ImageWithAttributesDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and their associated attributes from a CSV file.
    Supports both training/validation (with attributes) and test (images only) modes.
    """

    def __init__(self, csv_file=None, image_dir=None, transform=None, test_mode=False):
        """
        Args:
            csv_file (str, optional): Path to the CSV file containing image IDs and attributes.
            image_dir (str): Directory containing image files.
            transform (callable, optional): Optional transform to be applied on an image.
            test_mode (bool): If True, loads images without attributes (for test set).
        """
        self.image_dir = image_dir
        self.transform = transform
        self.test_mode = test_mode

        if not os.path.isdir(image_dir):
            raise ValueError(f"image_dir '{image_dir}' does not exist.")

        if not self.test_mode:
            # Training/validation mode: load CSV with IDs and attributes
            if csv_file is None:
                raise ValueError(
                    "csv_file must be provided in training/validation mode."
                )
            self.data = pd.read_csv(csv_file)
            self.ids = self.data["id"].astype(str).tolist()
            self.attributes = self.data.drop(columns="id").values.astype(float)
        else:
            # Test mode: only images, no CSV/attributes
            self.ids = None
            self.attributes = None
            self.all_images = [
                img for img in os.listdir(image_dir) if img.lower().endswith(".jpg")
            ]

        # Map image IDs to actual filenames (for training/validation)
        if not self.test_mode:
            all_images = os.listdir(image_dir)
            self.id_to_filename = {}
            for img_name in all_images:
                # Assumes filenames are like 'x123.jpg' where '123' is the ID
                if img_name.lower().endswith(".jpg") and img_name[1:-4] in self.ids:
                    self.id_to_filename[img_name[1:-4]] = img_name

    def __len__(self):
        """
        Returns:
            int: Number of samples in the dataset.
        """
        if self.test_mode:
            return len(self.all_images)
        else:
            return len(self.ids)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, attributes_tensor) in training/validation mode,
                   (image, filename) in test mode.
        """
        if self.test_mode:
            # Test mode: return image and filename
            img_name = self.all_images[idx]
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_name
        else:
            # Training/validation mode: return image and attributes
            img_id = self.ids[idx]
            img_name = self.id_to_filename.get(img_id)
            if not img_name:
                raise FileNotFoundError(f"Image not found for ID: {img_id}")
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            attributes_tensor = torch.tensor(self.attributes[idx], dtype=torch.float32)
            return image, attributes_tensor
