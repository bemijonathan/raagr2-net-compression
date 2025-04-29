import torch
import numpy as np
# Remove all TensorFlow and Keras imports


def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculate Dice coefficient

    Args:
        y_true: Ground truth tensor
        y_pred: Prediction tensor
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Mean Dice coefficient
    """
    # Ensure tensors are in the right format
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Calculate intersection and union
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3])

    # Calculate dice coefficient
    dice = torch.mean((2. * intersection + smooth)/(union + smooth), dim=0)
    return dice


'''
In case of binary IoU both functions below work exactly the same 
    i.e. the number of op_channel == 1
'''


def mean_iou(y_true, y_pred, smooth=1):
    """
    Calculate Mean IoU (Intersection over Union)

    Args:
        y_true: Ground truth tensor
        y_pred: Prediction tensor
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Mean IoU
    """
    # Ensure tensors are in the right format
    y_true = y_true.float()

    # Apply threshold to prediction
    if y_pred.dim() == 4:
        # If multi-channel output, take argmax along channel dimension
        if y_pred.shape[1] > 1:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True).float()
        else:
            y_pred = (y_pred > 0.5).float()

    # Print shapes for debugging and make necessary adjustments
    print(
        f"mean_iou: y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

    # Make sure dimensions match for multiplication
    if y_true.shape != y_pred.shape:
        # If y_true is [B, 1, H, W] and y_pred is [B, C, H, W], reshape y_pred
        if y_true.shape[1] == 1 and y_pred.shape[1] > 1:
            y_pred = y_pred[:, 0:1, :, :]
            print(f"Adjusted y_pred shape to: {y_pred.shape}")
        # If dimensions still don't match, try to fix the issue
        elif y_true.dim() == 3 and y_pred.dim() == 4:
            # Add channel dimension [B, H, W] -> [B, 1, H, W]
            y_true = y_true.unsqueeze(1)
            print(f"Adjusted y_true shape to: {y_true.shape}")

    # Calculate intersection and union
    try:
        intersection = torch.sum(
            y_true * y_pred, dim=[1, 2, 3] if y_true.dim() == 4 else [1, 2])
        union = torch.sum(y_true, dim=[1, 2, 3] if y_true.dim() == 4 else [1, 2]) + \
            torch.sum(y_pred, dim=[1, 2, 3] if y_pred.dim()
                      == 4 else [1, 2]) - intersection
    except Exception as e:
        print(f"Error calculating intersection and union: {e}")
        return 0.0

    # Calculate IoU
    iou = torch.mean((intersection + smooth) / (union + smooth))

    return iou
