"""Inference: generate predictions, compute metrics, save visualizations."""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

from scripts.postprocess import CRFPostProcess
from scripts.metrics import calculate_length_and_width_from_mask

IMG_WIDTH = 448
IMG_HEIGHT = 448


def predict_and_save(model, loader, device, cm_per_pixel_ratio, output_dir,
                     requires_length=True, postprocess_fn=None,
                     model_name="model"):
    """Run inference on *loader*, save comparison images and a CSV log.

    Returns
    -------
    pandas.DataFrame  with per-image metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, f"predicted_masks_{model_name}")
    os.makedirs(masks_dir, exist_ok=True)
    model.eval()
    results = []

    print(f"Running inference; results will be saved to {output_dir}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            images, masks_batch, lengths_batch, paths = batch
            if images.numel() == 0:
                continue
            images = images.to(device)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            gt_masks = masks_batch.numpy()
            gt_lengths = lengths_batch.numpy()
            imgs_np = images.cpu().numpy()

            for i in range(len(paths)):
                try:
                    entry = _process_sample(
                        paths[i], imgs_np[i], gt_masks[i], probs[i],
                        gt_lengths[i], cm_per_pixel_ratio, postprocess_fn,
                        requires_length, output_dir, masks_dir,
                    )
                    results.append(entry)
                except Exception as e:
                    print(f"Error processing {paths[i]}: {e}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "predictions_log.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Prediction log saved to: {csv_path}")

    if requires_length and len(results) > 0:
        _print_summary(df)

    return df


def _process_sample(img_path, img_np, gt_mask_np, pred_prob_np,
                    gt_length_np, ratio, postprocess_fn, requires_length,
                    output_dir, masks_dir):
    """Process a single image and return a log-entry dict."""
    basename = os.path.splitext(os.path.basename(img_path))[0]
    gt = gt_mask_np.squeeze()
    pred = pred_prob_np.squeeze()

    # Denormalise for visualisation
    img_viz = img_np.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_viz = np.clip((img_viz * std + mean) * 255, 0, 255).astype(np.uint8)

    use_img = img_viz if isinstance(postprocess_fn, CRFPostProcess) else None
    px_len, cleaned, skel, w_metrics = calculate_length_and_width_from_mask(
        pred, threshold=0.5, postprocess_fn=postprocess_fn,
        image=use_img, pixel_to_cm_ratio=ratio,
    )

    # Save predicted mask
    cv2.imwrite(
        os.path.join(masks_dir, f"{basename}_mask.png"),
        (cleaned * 255).astype(np.uint8),
    )

    # Per-image segmentation metrics
    pred_t = torch.from_numpy(cleaned.astype(int)).long()
    gt_t = torch.from_numpy((gt > 0.5).astype(int)).long()
    tp, fp, fn, tn = smp.metrics.get_stats(pred_t, gt_t, mode="binary")
    entry = {
        "filename": os.path.basename(img_path),
        "iou_score": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(),
        "f1_score": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(),
        "precision": smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item(),
        "recall": smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item(),
        "accuracy": smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item(),
    }

    if requires_length:
        true_cm = gt_length_np.item()
        _, _, _, gt_w = calculate_length_and_width_from_mask(
            gt, threshold=0.5, pixel_to_cm_ratio=ratio,
        )
        pred_cm = px_len * ratio
        err = abs(true_cm - pred_cm)
        err_rate = (err / true_cm * 100) if true_cm > 0 else 0.0
        entry.update({
            "true_length_cm": true_cm, "predicted_length_cm": pred_cm,
            "length_absolute_error_cm": err, "length_error_rate_%": err_rate,
            "true_mean_width_cm": gt_w["mean_width_cm"],
            "true_max_width_cm": gt_w["max_width_cm"],
            "true_median_width_cm": gt_w["median_width_cm"],
            "true_std_width_cm": gt_w["std_width_cm"],
            "pred_mean_width_cm": w_metrics["mean_width_cm"],
            "pred_max_width_cm": w_metrics["max_width_cm"],
            "pred_median_width_cm": w_metrics["median_width_cm"],
            "pred_std_width_cm": w_metrics["std_width_cm"],
            "width_mean_error_cm": abs(gt_w["mean_width_cm"] - w_metrics["mean_width_cm"]),
            "width_max_error_cm": abs(gt_w["max_width_cm"] - w_metrics["max_width_cm"]),
        })

    # Save side-by-side visualisation
    gt_viz = cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pred_viz = cv2.cvtColor((cleaned * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    h, w = IMG_HEIGHT, IMG_WIDTH
    cmp = np.hstack([
        cv2.resize(cv2.cvtColor(img_viz, cv2.COLOR_RGB2BGR), (w, h)),
        cv2.resize(gt_viz, (w, h)),
        cv2.resize(pred_viz, (w, h)),
    ])
    cv2.imwrite(os.path.join(output_dir, f"{basename}_pred.png"), cmp)
    return entry


def _print_summary(df):
    sep = "=" * 60
    print(f"\n{sep}\nPrediction Summary\n{sep}")
    if "iou_score" in df.columns:
        print("\n[Segmentation Metrics (per-image mean)]")
        for m in ("iou_score", "f1_score", "precision", "recall", "accuracy"):
            if m in df.columns:
                print(f"  {m}: {df[m].mean():.4f} (+/-{df[m].std():.4f})")
    if "length_absolute_error_cm" in df.columns:
        print("\n[Length Prediction]")
        print(f"  MAE: {df['length_absolute_error_cm'].mean():.4f} cm")
        print(f"  Mean error rate: {df['length_error_rate_%'].mean():.2f}%")
    if "width_mean_error_cm" in df.columns:
        print("\n[Width Prediction]")
        print(f"  Mean width MAE: {df['width_mean_error_cm'].mean():.4f} cm")
        print(f"  Max width MAE: {df['width_max_error_cm'].mean():.4f} cm")
    print(sep)
