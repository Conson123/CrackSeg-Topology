"""Loss functions for binary crack segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced segmentation."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        bce = F.binary_cross_entropy(pred_sigmoid, target, reduction="none")
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_weight * focal_weight
        return (focal_weight * bce).mean()


class TverskyLoss(nn.Module):
    """Tversky Loss with adjustable FP/FN weighting."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        tp = (pred_sigmoid * target).sum()
        fp = ((1 - target) * pred_sigmoid).sum()
        fn = (target * (1 - pred_sigmoid)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class LovaszHingeLoss(nn.Module):
    """Lovasz-Hinge Loss for direct IoU optimisation (binary)."""

    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, pred, target):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.float()
        if self.per_image:
            losses = []
            for i in range(pred.size(0)):
                losses.append(self._flat(pred[i:i + 1], target[i:i + 1]))
            return torch.stack(losses).mean()
        return self._flat(pred, target)

    @staticmethod
    def _flat(pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        signs = 2.0 * target - 1.0
        errors = 1.0 - pred * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        target_sorted = target[perm]
        inter = target_sorted.sum() - target_sorted.cumsum(0)
        union = target_sorted.sum() + (1.0 - target_sorted).cumsum(0)
        iou = 1.0 - inter / union
        if len(target_sorted) > 1:
            gts = torch.cat([iou[0:1], iou[1:] - iou[:-1]])
        else:
            gts = iou
        return (torch.relu(errors_sorted) * gts).sum()


class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax Loss."""

    def __init__(self, per_image=True):
        super().__init__()
        self.per_image = per_image

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        if pred_prob.dim() == 4:
            pred_prob = pred_prob.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.float()
        if self.per_image:
            losses = []
            for i in range(pred_prob.size(0)):
                losses.append(self._flat(pred_prob[i:i + 1], target[i:i + 1]))
            return torch.stack(losses).mean()
        return self._flat(pred_prob, target)

    @staticmethod
    def _flat(pred_prob, target):
        pred_prob = pred_prob.contiguous().view(-1)
        target = target.contiguous().view(-1)
        errors = torch.abs(target - pred_prob)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        target_sorted = target[perm]
        fg = target_sorted.sum()
        if fg == 0:
            return torch.tensor(0.0, device=pred_prob.device)
        inter = fg - target_sorted.cumsum(0)
        union = fg + (1.0 - target_sorted).cumsum(0)
        iou = 1.0 - inter / union
        if len(target_sorted) > 1:
            gts = torch.cat([iou[0:1], iou[1:] - iou[:-1]])
        else:
            gts = iou
        return (errors_sorted * gts).sum()


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss."""

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="binary")

    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + self.beta * self.dice(pred, target)


class FocalDiceLoss(nn.Module):
    """Focal + Dice combined loss."""

    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(mode="binary")
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (1 - self.dice_weight) * self.focal(pred, target) + \
               self.dice_weight * self.dice(pred, target)


class TverskyDiceLoss(nn.Module):
    """Tversky + Dice combined loss."""

    def __init__(self, tversky_alpha=0.3, tversky_beta=0.7, dice_weight=0.5):
        super().__init__()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.dice = DiceLoss(mode="binary")
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (1 - self.dice_weight) * self.tversky(pred, target) + \
               self.dice_weight * self.dice(pred, target)


class LovaszDiceLoss(nn.Module):
    """Lovasz + Dice combined loss."""

    def __init__(self, lovasz_type="hinge", dice_weight=0.5, per_image=True):
        super().__init__()
        if lovasz_type == "hinge":
            self.lovasz = LovaszHingeLoss(per_image=per_image)
        else:
            self.lovasz = LovaszSoftmaxLoss(per_image=per_image)
        self.dice = DiceLoss(mode="binary")
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (1 - self.dice_weight) * self.lovasz(pred, target) + \
               self.dice_weight * self.dice(pred, target)


class HybridLoss(nn.Module):
    """Multi-component loss: Dice + BCE + Tversky + Boundary."""

    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.dice_loss = DiceLoss(mode="binary")
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def _tversky(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        tp = (pred_sigmoid * target).sum()
        fp = ((1 - target) * pred_sigmoid).sum()
        fn = (target * (1 - pred_sigmoid)).sum()
        return 1 - (tp + 1) / (tp + self.alpha * fp + self.beta * fn + 1)

    def _boundary(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        try:
            dilated = F.conv2d(target, kernel, padding=1)
            eroded = F.conv2d(target, kernel, padding=1)
            boundary = ((dilated - eroded).abs() > 0).float()
            return F.binary_cross_entropy(
                pred_sigmoid * boundary + 1e-7, target * boundary + 1e-7
            )
        except Exception:
            return torch.tensor(0.0, device=pred.device)

    def forward(self, pred, target):
        return (0.4 * self.dice_loss(pred, target)
                + 0.2 * self.bce_loss(pred, target)
                + 0.3 * self._tversky(pred, target)
                + 0.1 * self._boundary(pred, target))


class DeepSupervisionLoss(nn.Module):
    """Weighted sum of main and auxiliary losses for deep supervision."""

    def __init__(self, base_loss, aux_weights=None):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights or [0.4, 0.3, 0.2]

    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main_out, aux_outs = outputs
            loss = self.base_loss(main_out, target)
            for i, aux_out in enumerate(aux_outs):
                if i < len(self.aux_weights):
                    loss += self.aux_weights[i] * self.base_loss(aux_out, target)
            return loss
        return self.base_loss(outputs, target)


def get_loss_function(loss_type="combined"):
    """Factory function returning the specified loss."""
    factories = {
        "combined": lambda: CombinedLoss(0.5, 0.5),
        "focal": lambda: FocalDiceLoss(0.25, 2.0, 0.5),
        "tversky": lambda: TverskyDiceLoss(0.3, 0.7, 0.5),
        "lovasz": lambda: LovaszDiceLoss("hinge", 0.5, True),
        "lovasz_softmax": lambda: LovaszDiceLoss("softmax", 0.5, True),
    }
    if loss_type not in factories:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return factories[loss_type]()
