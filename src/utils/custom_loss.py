import torch
import torch.nn as nn


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = torch.log(y_pred / (1. - y_pred))

    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
        (torch.log(1. + torch.exp(-torch.abs(logit_y_pred))) +
         torch.maximum(-logit_y_pred, torch.tensor(0.)))
    return torch.sum(loss) / torch.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight=1):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * torch.sum(w * intersection) + smooth) / (torch.sum(w * (m1**2)) +
                                                           # Uptill here is Dice Loss with squared
                                                           torch.sum(w * (m2**2)) + smooth)
    loss = 1. - torch.sum(score)  # Soft Dice Loss
    return loss


def Dice_coef(y_true, y_pred, weight=1.):
    smooth = 1.
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / \
        (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def Dice_loss(y_true, y_pred):
    loss = 1 - Dice_coef(y_true, y_pred)
    return loss


def Weighted_BCEnDice_loss(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()

    # if we want to get same size of output, kernel size must be odd number
    # PyTorch equivalent of tf.nn.avg_pool2d
    avg_pool = nn.AvgPool2d(kernel_size=11, stride=1, padding=5)
    averaged_mask = avg_pool(y_true)

    border = (averaged_mask > 0.005).float() * (averaged_mask < 0.995).float()
    weight = torch.ones_like(averaged_mask)
    w0 = torch.sum(weight)
    weight += border * 2
    w1 = torch.sum(weight)
    weight *= (w0 / w1)

    loss = weighted_dice_loss(y_true, y_pred, weight) + \
        weighted_bce_loss(y_true, y_pred, weight)
    return loss
