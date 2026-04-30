"""Crack geometric measurement: skeletonisation-based length and
EDT-based width estimation."""

import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, label
from scipy.ndimage import distance_transform_edt
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from models.width_estimator import CrackWidthEstimator

POST_PROCESS_MIN_SIZE = 50


def calculate_length_and_width_from_mask(
    mask, threshold=0.5, postprocess_fn=None, image=None, pixel_to_cm_ratio=1.0
):
    """Compute crack length (skeleton pixels) and width statistics.

    Returns
    -------
    pixel_length : int
    cleaned_mask : ndarray
    skeleton : ndarray
    width_metrics : dict
    """
    if mask.ndim == 3:
        mask = mask.squeeze()

    if postprocess_fn is not None:
        cleaned = (postprocess_fn(mask, image=image, threshold=threshold)
                   if image is not None
                   else postprocess_fn(mask, threshold=threshold))
    else:
        cleaned = remove_small_objects(
            mask > threshold, min_size=POST_PROCESS_MIN_SIZE, connectivity=1
        )

    skeleton = skeletonize(cleaned)
    pixel_length = int(np.sum(skeleton))

    estimator = CrackWidthEstimator(pixel_to_cm_ratio=pixel_to_cm_ratio)
    width_metrics = estimator.estimate_width(cleaned, skeleton=skeleton)

    return pixel_length, cleaned, skeleton, width_metrics


def calculate_length_from_mask(mask, threshold=0.5, postprocess_fn=None, image=None):
    """Light version returning only pixel length and cleaned mask."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    if postprocess_fn is not None:
        cleaned = (postprocess_fn(mask, image=image, threshold=threshold)
                   if image is not None
                   else postprocess_fn(mask, threshold=threshold))
    else:
        cleaned = remove_small_objects(
            mask > threshold, min_size=POST_PROCESS_MIN_SIZE, connectivity=1
        )
    skeleton = skeletonize(cleaned)
    return int(np.sum(skeleton)), cleaned


def improved_length_calculation(skeleton, pixel_to_cm_ratio=1.0):
    """MST-based length on connected skeleton components."""
    if skeleton.sum() == 0:
        return 0.0
    try:
        labeled = label(skeleton)
        total = 0.0
        for rid in range(1, labeled.max() + 1):
            coords = np.argwhere(labeled == rid)
            if len(coords) < 2:
                continue
            mst = minimum_spanning_tree(distance_matrix(coords, coords))
            total += mst.sum()
        return total * pixel_to_cm_ratio
    except Exception:
        return float(np.sum(skeleton)) * pixel_to_cm_ratio


def estimate_pixel_to_cm_ratio(dataset_info, transform):
    """Estimate global pixel-to-cm scale from ground-truth annotations."""
    from data.dataset import CrackDataset

    print("Estimating pixel-to-cm ratio ...")
    pixel_lengths, cm_lengths = [], []
    ds = CrackDataset(dataset_info, transform=transform)
    for i in range(len(ds)):
        _, mask_t, length_t, _ = ds[i]
        if mask_t is None:
            continue
        px_len, _ = calculate_length_from_mask(mask_t.squeeze().numpy())
        if px_len > 0:
            pixel_lengths.append(px_len)
            cm_lengths.append(length_t.item())

    if not pixel_lengths:
        print("Warning: no valid samples; returning default ratio 1.0")
        return 1.0

    ratio = sum(cm_lengths) / sum(pixel_lengths)
    print(f"Estimated ratio from {len(pixel_lengths)} samples: "
          f"1 px = {ratio:.6f} cm")
    return ratio
