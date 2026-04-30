"""Albumentations-based augmentation pipelines."""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(img_height=448, img_width=448):
    """Return (train_transform, val_transform) pair."""
    train = A.Compose([
        A.Resize(img_height, img_width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.8, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.ElasticTransform(p=0.5, alpha=120, sigma=6.0, alpha_affine=3.6,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.GridDistortion(p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    val = A.Compose([
        A.Resize(img_height, img_width),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train, val


def get_advanced_training_augmentation(img_height=448, img_width=448):
    """Heavier augmentation pipeline for pre-training."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Resize(img_height, img_width),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
