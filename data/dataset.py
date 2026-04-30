"""PyTorch Dataset for crack segmentation images and masks."""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    """Load image/mask pairs with optional crack-length labels.

    Parameters
    ----------
    dataset_info : array-like of dicts
        Each entry must contain keys ``"image"`` and ``"mask"`` (file paths)
        and optionally ``"length"`` (ground-truth crack length in cm).
    transform : albumentations.Compose or None
        Augmentation / preprocessing pipeline applied to both image and mask.
    """

    def __init__(self, dataset_info, transform=None):
        self.dataset_info = dataset_info
        self.transform = transform

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, idx):
        info = self.dataset_info[idx]
        try:
            img = cv2.imread(info["image"])
            if img is None:
                raise IOError(f"Cannot read image: {info['image']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(info["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(f"Cannot read mask: {info['mask']}")
            mask = (mask > 0).astype(np.float32)[..., np.newaxis]

            length = np.array([info.get("length", 0.0)], dtype=np.float32)

            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"].permute(2, 0, 1)

            return img, mask, length, info["image"]
        except Exception:
            return None, None, None, None


def collate_fn(batch):
    """Filter out failed samples before collation."""
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), []
    return torch.utils.data.dataloader.default_collate(batch)
