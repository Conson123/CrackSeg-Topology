"""Post-processing strategies for binary segmentation masks."""

import numpy as np
from skimage.morphology import remove_small_objects, binary_dilation, binary_erosion

POST_PROCESS_MIN_SIZE = 50


class MorphologicalPostProcess:
    """Opening + closing + small-object removal."""

    def __init__(self, min_size=50, closing_kernel=5, opening_kernel=3):
        self.min_size = min_size
        self.closing_kernel = closing_kernel
        self.opening_kernel = opening_kernel

    def __call__(self, mask, image=None, threshold=0.5):
        binary = mask > threshold
        if self.opening_kernel > 0:
            k = np.ones((self.opening_kernel, self.opening_kernel), np.uint8)
            binary = binary_dilation(binary_erosion(binary, k), k)
        if self.closing_kernel > 0:
            k = np.ones((self.closing_kernel, self.closing_kernel), np.uint8)
            binary = binary_erosion(binary_dilation(binary, k), k)
        if self.min_size > 0:
            binary = remove_small_objects(binary, min_size=self.min_size, connectivity=1)
        return binary


class CRFPostProcess:
    """Dense CRF refinement (requires ``pydensecrf``)."""

    def __init__(self, iter_max=10, pos_w=3, pos_xy_std=3,
                 bi_w=5, bi_xy_std=50, bi_rgb_std=5):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
            self.dcrf = dcrf
            self.unary_from_softmax = unary_from_softmax
            self.available = True
        except ImportError:
            print("Warning: pydensecrf not installed; "
                  "CRF post-processing falls back to morphological.")
            self.available = False
            self.fallback = MorphologicalPostProcess()

    def __call__(self, mask, image=None, threshold=0.5):
        if not self.available or image is None:
            if hasattr(self, "fallback"):
                return self.fallback(mask, threshold=threshold)
            return mask > threshold

        h, w = mask.shape
        prob_fg = np.clip(mask, 1e-8, 1 - 1e-8)
        probs = np.stack([1 - prob_fg, prob_fg], axis=0)
        unary = np.ascontiguousarray(self.unary_from_softmax(probs))

        d = self.dcrf.DenseCRF2D(w, h, 2)
        d.setUnaryEnergy(unary)
        if image is not None:
            d.addPairwiseBilateral(
                sxy=self.bi_xy_std, srgb=self.bi_rgb_std,
                rgbim=np.ascontiguousarray(image.reshape(-1, 3).T),
                compat=self.bi_w,
            )
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        q = d.inference(self.iter_max)
        return np.argmax(q, axis=0).reshape(h, w).astype(bool)


def get_postprocess_function(postprocess_type="basic", **kwargs):
    """Factory returning the requested post-processing callable."""
    if postprocess_type == "basic":
        def _basic(mask, image=None, threshold=0.5):
            return remove_small_objects(
                mask > threshold, min_size=POST_PROCESS_MIN_SIZE, connectivity=1
            )
        return _basic
    if postprocess_type == "morphological":
        return MorphologicalPostProcess(**kwargs)
    if postprocess_type == "crf":
        return CRFPostProcess(**kwargs)
    raise ValueError(f"Unknown post-process type: {postprocess_type}")
