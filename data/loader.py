"""Functions to discover and pair image/mask files for pre-training and
fine-tuning datasets."""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data_paths_finetune(data_dir):
    """Load image/mask/length triplets for the fine-tuning (SUT-Crack) dataset.

    Expects:
      ``<data_dir>/images/*.jpg``
      ``<data_dir>/labels/<basename>.png``
      ``<data_dir>/Crack Length.txt``  (tab-separated with *filename* and
      *Crack Length (cm)* columns)

    Returns
    -------
    numpy.ndarray of dicts or None on failure.
    """
    print(f"Loading fine-tuning data from {data_dir} ...")
    length_file = os.path.join(data_dir, "Crack Length.txt")
    try:
        length_df = pd.read_csv(length_file, sep="\t")
        length_map = {
            str(row["filename"]): row["Crack Length (cm)"]
            for _, row in length_df.iterrows()
        }
    except Exception as e:
        print(f"Error: cannot load {length_file}. {e}")
        return None

    image_paths = glob.glob(os.path.join(data_dir, "images", "*.jpg"))
    if not image_paths:
        print(f"Error: no .jpg images found in {data_dir}/images/.")
        return None

    info = []
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(data_dir, "labels", f"{basename}.png")
        if os.path.exists(mask_path) and basename in length_map:
            info.append({"image": img_path, "mask": mask_path,
                         "length": length_map[basename]})
        else:
            print(f"Warning: skipping {basename} (missing mask or length).")

    print(f"Matched {len(info)} fine-tuning samples.")
    return np.array(info)


def load_data_paths_pretrain(data_dir):
    """Load image/mask pairs for pre-training (no length labels).

    Expects ``train/images``, ``train/masks``, ``test/images``, ``test/masks``
    sub-directories under *data_dir*.

    Returns
    -------
    (train_info, val_info) : tuple of numpy.ndarray
    """
    print(f"Loading pre-training data from {data_dir} ...")

    def _collect(split):
        img_dir = os.path.join(data_dir, split, "images")
        mask_dir = os.path.join(data_dir, split, "masks")
        paths = glob.glob(os.path.join(img_dir, "*.jpg"))
        paths += glob.glob(os.path.join(img_dir, "*.png"))
        entries = []
        for img_path in tqdm(paths, desc=f"Loading pre-train ({split})"):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, basename + ".png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, os.path.basename(img_path))
            if os.path.exists(mask_path):
                entries.append({"image": img_path, "mask": mask_path, "length": 0.0})
        return np.array(entries)

    train_info = _collect("train")
    val_info = _collect("test")
    print(f"Pre-training data: {len(train_info)} train, {len(val_info)} val samples.")
    return train_info, val_info
