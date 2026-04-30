"""Training and validation loops."""

import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm

from scripts.metrics import calculate_length_from_mask


def train_epoch(model, loader, optimizer, loss_fn, device, scheduler=None):
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0
    for images, masks, _, _ in tqdm(loader, desc="Training"):
        if images.numel() == 0:
            continue
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), masks)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def validate_epoch(model, loader, loss_fn, metrics_dict, device,
                   postprocess_fn=None):
    """Run one validation epoch; return (mean_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    tp_sum = torch.tensor(0.0, device=device)
    fp_sum = torch.tensor(0.0, device=device)
    fn_sum = torch.tensor(0.0, device=device)
    tn_sum = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for images, masks, _, _ in tqdm(loader, desc="Validating"):
            if images.numel() == 0:
                continue
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            total_loss += loss_fn(preds, masks).item()

            probs = torch.sigmoid(preds)
            gt = masks.long()
            for i in range(probs.shape[0]):
                pred_np = probs[i].squeeze().cpu().numpy()
                gt_np = gt[i].squeeze().cpu().numpy()
                _, cleaned = calculate_length_from_mask(
                    pred_np, threshold=0.5, postprocess_fn=postprocess_fn
                )
                pred_t = torch.from_numpy(cleaned.astype(int)).long().to(device)
                gt_t = torch.from_numpy(gt_np).long().to(device)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_t, gt_t, mode="binary")
                tp_sum += tp.sum()
                fp_sum += fp.sum()
                fn_sum += fn.sum()
                tn_sum += tn.sum()

    results = {
        name: fn(tp_sum, fp_sum, fn_sum, tn_sum, reduction="micro").item()
        for name, fn in metrics_dict.items()
    }
    return total_loss / max(len(loader), 1), results
