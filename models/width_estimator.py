"""Crack width estimation via Euclidean Distance Transform on skeleton."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


class CrackWidthEstimator:
    """Estimate physical crack width from a binary segmentation mask.

    The primary method is distance-transform sampling along the morphological
    skeleton: for every skeleton pixel *p*, the EDT value gives the shortest
    distance to the nearest background pixel (i.e. crack half-width), so
    ``width(p) = 2 * EDT(p)``.

    Parameters
    ----------
    pixel_to_cm_ratio : float
        Scale factor converting pixels to centimetres (default 1.0).
    """

    def __init__(self, pixel_to_cm_ratio=1.0):
        self.pixel_to_cm_ratio = pixel_to_cm_ratio

    def estimate_width(self, mask, skeleton=None):
        """Return a dict of width statistics (pixels and cm).

        Parameters
        ----------
        mask : ndarray, shape (H, W) or (H, W, 1)
            Binary segmentation mask.
        skeleton : ndarray, optional
            Pre-computed skeleton; computed automatically if not supplied.
        """
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = (mask > 0.5).astype(np.uint8)

        empty = {
            "mean_width_px": 0.0, "mean_width_cm": 0.0,
            "max_width_px": 0.0, "max_width_cm": 0.0,
            "median_width_px": 0.0, "median_width_cm": 0.0,
            "std_width_px": 0.0, "std_width_cm": 0.0,
            "width_distribution": [],
        }
        if binary.sum() == 0:
            return empty

        dt = distance_transform_edt(binary)
        if skeleton is None:
            skeleton = skeletonize(binary > 0)

        coords = np.where(skeleton)
        if len(coords[0]) == 0:
            widths = dt[binary > 0] * 2
        else:
            widths = dt[coords] * 2
        widths = widths[widths > 0]

        if len(widths) == 0:
            return empty

        r = self.pixel_to_cm_ratio
        return {
            "mean_width_px": float(np.mean(widths)),
            "mean_width_cm": float(np.mean(widths) * r),
            "max_width_px": float(np.max(widths)),
            "max_width_cm": float(np.max(widths) * r),
            "median_width_px": float(np.median(widths)),
            "median_width_cm": float(np.median(widths) * r),
            "std_width_px": float(np.std(widths)),
            "std_width_cm": float(np.std(widths) * r),
            "width_distribution": widths.tolist(),
        }

    def visualize_width_distribution(self, mask, save_path=None):
        """Plot mask, EDT heatmap, skeleton overlay, and width histogram."""
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        binary = (mask > 0.5).astype(np.uint8)
        if binary.sum() == 0:
            print("Warning: empty mask, nothing to visualize.")
            return

        dt = distance_transform_edt(binary)
        skeleton = skeletonize(binary > 0)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(binary, cmap="gray")
        axes[0, 0].set_title("Binary Mask")
        axes[0, 0].axis("off")

        im = axes[0, 1].imshow(dt, cmap="hot")
        axes[0, 1].set_title("Distance Transform (half-width)")
        axes[0, 1].axis("off")
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

        overlay = np.zeros((*binary.shape, 3))
        overlay[:, :, 0] = dt / (dt.max() + 1e-8)
        overlay[:, :, 1] = skeleton.astype(float)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title("Skeleton + Distance Map")
        axes[1, 0].axis("off")

        coords = np.where(skeleton)
        if len(coords[0]) > 0:
            widths = dt[coords] * 2
            widths = widths[widths > 0]
            axes[1, 1].hist(widths, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
            axes[1, 1].axvline(
                np.mean(widths), color="red", linestyle="--",
                label=f"Mean: {np.mean(widths):.2f} px", linewidth=2,
            )
            axes[1, 1].set_title("Width Distribution")
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Width distribution figure saved to: {save_path}")
        else:
            plt.show()
        plt.close()
