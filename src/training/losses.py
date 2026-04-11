import torch
import torch.nn.functional as F

from configs.config import Config

# Weights tuned for actual pixel distribution in your dataset:
#   bg_only=0%  → no pure background slices (data.py already filtered)
#   has_ncr  60.7% of slices  → class 1 moderate weight
#   has_edema 99.8% of slices → class 2 already common, moderate weight
#   has_et   72.1% of slices  → class 3 needs slight boost
#
# At pixel level ET is still smallest, but far better represented than before.
# Equal foreground weights (2.0) as starting point — adjust after first run.
MULTICLASS_WEIGHTS = [0.05, 2.0, 2.0, 2.5]


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f     = y_true.reshape(-1)
    y_pred_f     = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def iou_coef(y_true, y_pred, smooth=1.0):
    y_true_f     = y_true.reshape(-1)
    y_pred_f     = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union        = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    y_pred       = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
    bce          = -(y_true * torch.log(y_pred) +
                     (1.0 - y_true) * torch.log(1.0 - y_pred))
    weighted_bce = bce * (y_true * 5.0 + 1.0)
    return weighted_bce.mean() + dice_loss(y_true, y_pred)


def multiclass_dice_coef(y_true_long, y_pred_softmax,
                          num_classes=Config.NUM_CLASSES, smooth=1e-6):
    scores = []
    for c in range(1, num_classes):
        true_c   = (y_true_long == c).float()
        pred_c   = y_pred_softmax[:, c, :, :]
        true_sum = true_c.sum()
        pred_sum = pred_c.sum()
        if true_sum == 0 and pred_sum < 0.1:
            continue
        intersection = (true_c * pred_c).sum()
        scores.append((2.0 * intersection + smooth) / (true_sum + pred_sum + smooth))
    if not scores:
        return torch.tensor(0.0, device=y_pred_softmax.device)
    return torch.stack(scores).mean()


def multiclass_iou_coef(y_true_long, y_pred_softmax,
                         num_classes=Config.NUM_CLASSES, smooth=1e-6):
    scores = []
    for c in range(1, num_classes):
        true_c   = (y_true_long == c).float()
        pred_c   = y_pred_softmax[:, c, :, :]
        true_sum = true_c.sum()
        pred_sum = pred_c.sum()
        if true_sum == 0 and pred_sum < 0.1:
            continue
        intersection = (true_c * pred_c).sum()
        union        = true_sum + pred_sum - intersection
        scores.append((intersection + smooth) / (union + smooth))
    if not scores:
        return torch.tensor(0.0, device=y_pred_softmax.device)
    return torch.stack(scores).mean()


def per_class_dice(y_true_long, y_pred_softmax,
                   num_classes=Config.NUM_CLASSES, smooth=1e-6):
    scores = []
    for c in range(1, num_classes):
        true_c   = (y_true_long == c).float()
        pred_c   = y_pred_softmax[:, c, :, :]
        true_sum = true_c.sum()
        pred_sum = pred_c.sum()
        if true_sum == 0 and pred_sum < 0.1:
            scores.append(0.0)
            continue
        intersection = (true_c * pred_c).sum()
        scores.append(((2.0 * intersection + smooth) /
                       (true_sum + pred_sum + smooth)).item())
    return scores


def per_class_iou(y_true_long, y_pred_softmax,
                  num_classes=Config.NUM_CLASSES, smooth=1e-6):
    scores = []
    for c in range(1, num_classes):
        true_c   = (y_true_long == c).float()
        pred_c   = y_pred_softmax[:, c, :, :]
        true_sum = true_c.sum()
        pred_sum = pred_c.sum()
        if true_sum == 0 and pred_sum < 0.1:
            scores.append(0.0)
            continue
        intersection = (true_c * pred_c).sum()
        union        = true_sum + pred_sum - intersection
        scores.append(((intersection + smooth) / (union + smooth)).item())
    return scores


def multiclass_dice_loss(y_true_long, y_pred_softmax,
                          num_classes=Config.NUM_CLASSES):
    return 1.0 - multiclass_dice_coef(y_true_long, y_pred_softmax, num_classes)


def focal_dice_loss(y_true_long, y_pred_softmax,
                    num_classes=Config.NUM_CLASSES, smooth=1e-6, gamma=2.0):
    scores = []
    for c in range(1, num_classes):
        true_c   = (y_true_long == c).float()
        pred_c   = y_pred_softmax[:, c, :, :]
        true_sum = true_c.sum()
        pred_sum = pred_c.sum()
        if true_sum == 0 and pred_sum < 0.1:
            continue
        intersection = (true_c * pred_c).sum()
        dice_c       = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
        focal_w      = (1.0 - dice_c) ** gamma
        scores.append(focal_w * (1.0 - dice_c))
    if not scores:
        return torch.tensor(0.0, device=y_pred_softmax.device)
    return torch.stack(scores).mean()


def ce_dice_loss(y_true_long, y_pred_logits, num_classes=Config.NUM_CLASSES):
    weights = torch.tensor(MULTICLASS_WEIGHTS, dtype=torch.float32,
                           device=y_pred_logits.device)
    ce      = F.cross_entropy(y_pred_logits, y_true_long, weight=weights)
    fdl     = focal_dice_loss(y_true_long, F.softmax(y_pred_logits, dim=1), num_classes)
    return ce + fdl