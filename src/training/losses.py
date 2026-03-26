# src/training/losses.py
import torch
import torch.nn as nn


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def iou_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
    bce = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    weight = 5.0
    weighted_bce = bce * (y_true * weight + 1.0)
    return weighted_bce.mean() + dice_loss(y_true, y_pred)