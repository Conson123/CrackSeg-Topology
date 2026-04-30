#!/usr/bin/env python3
"""TopoCrackSeg: Topology-Aware Crack Segmentation via Multi-Source
Transfer Learning for Physical Dimension Quantification.

Two-stage training pipeline (pre-training + fine-tuning) with crack length
and width estimation, multiple loss functions, attention mechanisms,
ASPP multi-scale context, and multi-GPU support.

Usage examples
--------------
# Stage 1: Pre-train on a source dataset
python main.py --mode pretrain --data_dir data/CFD --dataset_name CFD

# Stage 2: Fine-tune on SUT-Crack with pre-trained weights
python main.py --mode finetune --data_dir data/SUT-Crack \\
    --load_weights run_results/pretrain_CFD_UnetPlusPlus_mobilenet_v2/pretrained_UnetPlusPlus_mobilenet_v2.pth

# Analyse results across experiments
python main.py --mode analyze
"""

__version__ = "1.0.0"
__author__ = "Pinglang Kou"

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from data import CrackDataset, collate_fn, get_transforms
from models import get_model
from models.network import save_model_clean
from models.losses import get_loss_function
from scripts.postprocess import get_postprocess_function
from scripts.train import train_epoch, validate_epoch
from scripts.predict import predict_and_save
from scripts.metrics import estimate_pixel_to_cm_ratio
from scripts.analyze import run_analysis

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
IMG_HEIGHT = 448
IMG_WIDTH = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

METRICS = {
    "iou_score": smp.metrics.iou_score,
    "f1_score": smp.metrics.f1_score,
    "precision": smp.metrics.precision,
    "recall": smp.metrics.recall,
    "accuracy": smp.metrics.accuracy,
}


def _save_training_curves(log_df, save_dir, run_name):
    """Save loss and metric curves to PNG."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        plt.figure(figsize=(18, 6))
        sns.set(style="whitegrid")
        plt.subplot(1, 2, 1)
        sns.lineplot(x=log_df["epoch"], y=log_df["train_loss"], label="Train Loss")
        sns.lineplot(x=log_df["epoch"], y=log_df["val_loss"], label="Val Loss")
        plt.title(f"{run_name} - Loss Curve")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.subplot(1, 2, 2)
        for col in ("val_iou_score", "val_f1_score", "val_precision",
                     "val_recall", "val_accuracy"):
            if col in log_df.columns:
                sns.lineplot(x=log_df["epoch"], y=log_df[col], label=col)
        plt.title(f"{run_name} - Validation Metrics")
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.ylim(0, 1); plt.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{run_name}_curves.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Training curves saved to: {path}")
    except Exception as e:
        print(f"Warning: failed to plot curves ({e})")


# ====================================================================
# Stage 1: Pre-training
# ====================================================================

def run_pretraining(args, base_dir):
    from data.loader import load_data_paths_pretrain

    print("--- [Stage 1: Pre-training] ---")
    save_path = os.path.join(base_dir, f"pretrained_{args.model}_{args.encoder}.pth")
    log_file = os.path.join(base_dir, "pretrain_log.csv")

    train_info, val_info = load_data_paths_pretrain(args.data_dir)
    if train_info is None or len(train_info) == 0:
        print("Pre-training data loading failed. Aborting.")
        return

    train_tf, val_tf = get_transforms(IMG_HEIGHT, IMG_WIDTH)
    train_loader = DataLoader(CrackDataset(train_info, train_tf),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(CrackDataset(val_info, val_tf),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    model = get_model(args.model, args.encoder, "imagenet",
                      attention_type=args.attention,
                      use_aspp=args.use_aspp, device=DEVICE).to(DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs detected")
        model = nn.DataParallel(model)

    loss_fn = get_loss_function(args.loss_type)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    pp_fn = get_postprocess_function(args.postprocess)

    best_iou, no_improve, logs = 0.0, 0, []
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- (Pre-train) Epoch {epoch}/{args.epochs} ---")
        t_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        v_loss, v_met = validate_epoch(model, val_loader, loss_fn, METRICS, DEVICE, pp_fn)
        print(f"Epoch {epoch}: train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in v_met.items()))

        entry = {"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss}
        entry.update({f"val_{k}": v for k, v in v_met.items()})
        logs.append(entry)

        iou = v_met["iou_score"]
        if iou > best_iou:
            best_iou = iou
            save_model_clean(model, save_path)
            print(f"  New best IoU={best_iou:.4f}. Model saved.")
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{args.patience}).")
        if no_improve >= args.patience:
            print(f"Early stopping triggered after {args.patience} epochs.")
            break

    log_df = pd.DataFrame(logs)
    log_df.to_csv(log_file, index=False, float_format="%.6f")
    _save_training_curves(log_df, base_dir, "pretrain")
    print("--- [Stage 1] Complete ---")


# ====================================================================
# Stage 2: Fine-tuning (K-Fold)
# ====================================================================

def run_finetuning(args, base_dir):
    from data.loader import load_data_paths_finetune

    print("--- [Stage 2: Fine-tuning (K-Fold)] ---")
    summary_file = os.path.join(base_dir, "kfold_summary.csv")
    dataset_info = load_data_paths_finetune(args.data_dir)
    if dataset_info is None or len(dataset_info) == 0:
        print("Fine-tuning data loading failed. Aborting.")
        return

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=SEED)
    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_info)):
        fold_num = fold + 1
        print(f"\n=== Fold {fold_num}/{args.k_folds} ===")
        train_info, val_info = dataset_info[train_idx], dataset_info[val_idx]
        print(f"  Train: {len(train_info)}  Val: {len(val_info)}")

        train_tf, val_tf = get_transforms(IMG_HEIGHT, IMG_WIDTH)
        train_loader = DataLoader(CrackDataset(train_info, train_tf),
                                  batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2)
        val_loader = DataLoader(CrackDataset(val_info, val_tf),
                                batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=2)

        ratio = estimate_pixel_to_cm_ratio(train_info, val_tf)

        model = get_model(args.model, args.encoder, encoder_weights=None,
                          load_weights_path=args.load_weights,
                          attention_type=args.attention,
                          use_aspp=args.use_aspp, device=DEVICE).to(DEVICE)
        if torch.cuda.device_count() > 1:
            print(f"  Multi-GPU: {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        loss_fn = get_loss_function(args.loss_type)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        pp_fn = get_postprocess_function(args.postprocess)

        model_path = os.path.join(base_dir, f"model_fold_{fold_num}.pth")
        fold_log_file = os.path.join(base_dir, f"log_fold_{fold_num}.csv")
        fold_results_dir = os.path.join(base_dir, f"results_fold_{fold_num}")
        best_iou, no_improve, logs, best_entry = 0.0, 0, [], {}

        for epoch in range(1, args.epochs + 1):
            t_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            v_loss, v_met = validate_epoch(model, val_loader, loss_fn, METRICS, DEVICE, pp_fn)
            print(f"  Fold {fold_num} Epoch {epoch}: "
                  f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  "
                  f"IoU={v_met['iou_score']:.4f}")

            entry = {"epoch": epoch, "train_loss": t_loss, "val_loss": v_loss}
            entry.update({f"val_{k}": v for k, v in v_met.items()})
            logs.append(entry)

            iou = v_met["iou_score"]
            if iou > best_iou:
                best_iou = iou
                save_model_clean(model, model_path)
                best_entry = entry.copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        # Save training log and curves
        log_df = pd.DataFrame(logs)
        log_df.to_csv(fold_log_file, index=False, float_format="%.6f")
        _save_training_curves(log_df, base_dir, f"fold_{fold_num}")

        # Predict on validation set with best model
        try:
            sd = torch.load(model_path, map_location=DEVICE)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(sd)
            else:
                model.load_state_dict(sd)

            m_name = f"{args.model}_{args.encoder}_fold{fold_num}"
            pred_df = predict_and_save(
                model, val_loader, DEVICE, ratio, fold_results_dir,
                requires_length=True, postprocess_fn=pp_fn, model_name=m_name,
            )
            best_entry["val_mae_cm"] = pred_df["length_absolute_error_cm"].mean()
            if "length_error_rate_%" in pred_df.columns:
                best_entry["val_length_error_rate_%"] = pred_df["length_error_rate_%"].mean()
            if "width_mean_error_cm" in pred_df.columns:
                best_entry["val_mean_width_mae_cm"] = pred_df["width_mean_error_cm"].mean()
                best_entry["val_max_width_mae_cm"] = pred_df["width_max_error_cm"].mean()
            all_fold_metrics.append(best_entry)
        except Exception as e:
            print(f"  Prediction failed for fold {fold_num}: {e}")

    # K-fold summary
    if all_fold_metrics:
        sdf = pd.DataFrame(all_fold_metrics)
        mean_row = sdf.mean(); mean_row.name = "Mean"
        std_row = sdf.std();   std_row.name = "Std"
        full = pd.concat([sdf, mean_row.to_frame().T, std_row.to_frame().T])
        full.to_csv(summary_file, index=True, float_format="%.6f")
        print(f"\nK-Fold summary saved to: {summary_file}")
        print(full.to_string(float_format=lambda x: f"{x:.4f}"))
    print("--- [Stage 2] Complete ---")


# ====================================================================
# CLI
# ====================================================================

def main():
    p = argparse.ArgumentParser(
        description="TopoCrackSeg: crack segmentation and physical measurement"
    )
    p.add_argument("--mode", required=True,
                   choices=["pretrain", "finetune", "analyze"],
                   help="Run mode: pretrain, finetune, or analyze")
    p.add_argument("--data_dir", default=None, help="Dataset root directory")
    p.add_argument("--dataset_name", default=None,
                   help="Dataset tag used in output directory naming")
    p.add_argument("--model", default="UnetPlusPlus",
                   choices=["Unet", "Linknet", "FPN", "DeepLabV3",
                            "UnetPlusPlus", "MAnet"],
                   help="Segmentation architecture (default: UnetPlusPlus)")
    p.add_argument("--encoder", default="mobilenet_v2",
                   help="Encoder backbone (default: mobilenet_v2)")
    p.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--patience", type=int, default=200,
                   help="Early stopping patience (epochs)")
    p.add_argument("--load_weights", default=None,
                   help="Path to pre-trained .pth checkpoint")
    p.add_argument("--k_folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--loss_type", default="combined",
                   choices=["combined", "focal", "tversky", "lovasz",
                            "lovasz_softmax"],
                   help="Loss function (default: combined)")
    p.add_argument("--attention", default=None,
                   choices=[None, "cbam", "se", "eca"],
                   help="Attention mechanism (default: None)")
    p.add_argument("--use_aspp", action="store_true",
                   help="Insert ASPP at encoder bottleneck")
    p.add_argument("--postprocess", default="basic",
                   choices=["basic", "morphological", "crf"],
                   help="Post-processing strategy (default: basic)")
    args = p.parse_args()

    if args.mode == "analyze":
        run_analysis()
        return
    if args.data_dir is None:
        print("Error: --data_dir is required for pretrain/finetune modes.")
        return

    print(f"Device: {DEVICE}")
    print(f"Config: model={args.model}  encoder={args.encoder}  "
          f"loss={args.loss_type}  attention={args.attention}  "
          f"aspp={args.use_aspp}  postprocess={args.postprocess}")

    run_id = f"{args.mode}_{args.model}_{args.encoder}"
    if args.dataset_name:
        run_id = f"{args.mode}_{args.dataset_name}_{args.model}_{args.encoder}"
    if args.loss_type != "combined":
        run_id += f"_loss_{args.loss_type}"
    if args.attention:
        run_id += f"_att_{args.attention}"
    if args.postprocess != "basic":
        run_id += f"_post_{args.postprocess}"

    base_dir = os.path.join("run_results", run_id)
    os.makedirs(base_dir, exist_ok=True)
    print(f"Output directory: {base_dir}\n")

    if args.mode == "pretrain":
        run_pretraining(args, base_dir)
    elif args.mode == "finetune":
        if args.load_weights and args.load_weights.lower() == "none":
            args.load_weights = None
        run_finetuning(args, base_dir)


if __name__ == "__main__":
    main()
