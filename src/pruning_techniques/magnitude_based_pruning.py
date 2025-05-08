"""
Magnitude-Based Pruning for Neural Networks

This script implements magnitude-based pruning for neural network compression,
based on the approach described in the following research papers:

1. "Learning both Weights and Connections for Efficient Neural Networks"
   by Song Han, Jeff Pool, John Tran, and William J. Dally (2015)
   https://arxiv.org/abs/1506.02626

2. "To prune, or not to prune: exploring the efficacy of pruning for model compression"
   by Michael Zhu and Suyog Gupta (2017)
   https://arxiv.org/abs/1710.01878

Magnitude-based pruning removes parameters based on their absolute magnitude,
operating on the principle that smaller weights contribute less to the final
output and can be removed with minimal impact on model performance.

This implementation applies channel/filter pruning in convolutional neural networks
by calculating the L2-norm of each filter and removing those with the smallest magnitudes.
"""

import torch
import torch.nn as nn
import numpy as np
from utils.stats import ModelPerformanceMetrics, ModelComparison
from utils.load_data import BrainDataset
from torch.utils.data import DataLoader
from architecture.model import load_trained_model, class_dice
from utils.custom_metric import dice_coef, mean_iou
import os
import json
import copy
from src.CONFIGS import batch_size

# Set device to Metal Performance Shaders (MPS) for accelerated computation on Mac
device = torch.device("mps")


def get_model_size(model):
    """
    Calculate and return the number of trainable parameters in the model.

    Args:
        model: PyTorch model

    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_weight_magnitudes(model):
    """
    Calculate the magnitude of weights for each parameter in the model.

    For each convolutional layer, we compute the L2-norm of each filter,
    which serves as a measure of its importance. Filters with smaller
    norms are considered less important and candidates for pruning.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary mapping parameter names to their magnitude tensors
    """
    magnitudes = {}
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:  # Consider only weight matrices (not biases)
            # Calculate L2 norm across each filter (first dimension of weights)
            magnitudes[name] = torch.norm(
                param.data.view(param.size(0), -1), p=2, dim=1)
    return magnitudes


def create_pruning_mask(magnitudes, threshold_percentile):
    """
    Create pruning masks for each layer based on magnitude thresholds.

    We use percentile-based thresholding where weights below a certain
    percentile are pruned. This allows for more controlled pruning compared
    to using a fixed threshold value.

    Args:
        magnitudes: Dictionary mapping parameter names to their magnitude tensors
        threshold_percentile: Percentile threshold for pruning (0-100)

    Returns:
        dict: Dictionary of binary masks for each parameter
    """
    masks = {}
    for name, magnitude in magnitudes.items():
        # Compute percentile threshold
        threshold = np.percentile(
            magnitude.cpu().numpy(), threshold_percentile)
        # Create binary mask: 1 for weights to keep, 0 for weights to prune
        masks[name] = (magnitude > threshold).float()
    return masks


def apply_pruning_masks(model, masks):
    """
    Apply pruning masks to model parameters.

    This function zeroes out weights according to the pruning masks.
    Note that this is unstructured pruning that creates sparse matrices
    rather than removing neurons/channels entirely.

    Args:
        model: PyTorch model
        masks: Dictionary of binary masks for each parameter

    Returns:
        model: Pruned PyTorch model
    """
    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name]
            # Properly reshape mask to match parameter dimensions
            # For a conv weight of shape [out_channels, in_channels, kernel_h, kernel_w]
            # We expand our mask of shape [out_channels] to match
            reshaped_mask = mask
            for i in range(1, param.dim()):
                reshaped_mask = reshaped_mask.unsqueeze(-1)
            expanded_mask = reshaped_mask.expand_as(param.data)
            # Apply mask - zero out weights below threshold
            param.data.mul_(expanded_mask)
    return model


def evaluate_model(model, val_loader, batch_size=8):
    """
    Evaluate model on validation data.

    Computes multiple metrics relevant for medical image segmentation:
    - Mean IoU (Intersection over Union)
    - Dice coefficient (overall)
    - Class-specific Dice coefficients for each segmentation class

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        batch_size: Batch size for evaluation

    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    total_iou = 0
    total_dice = 0
    total_class_dice = {2: 0, 3: 0, 4: 0}
    num_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            # Calculate metrics
            iou = mean_iou(outputs, masks)
            dice = dice_coef(outputs, masks)
            class_dices = {
                # Class 2 (typically NCR - necrotic tumor core)
                2: class_dice(outputs, masks, 2),
                # Class 3 (typically ET - enhancing tumor)
                3: class_dice(outputs, masks, 3),
                # Class 4 (typically ED - peritumoral edema)
                4: class_dice(outputs, masks, 4)
            }

            total_iou += iou.item()
            total_dice += dice.item()
            for i in [2, 3, 4]:
                total_class_dice[i] += class_dices[i].item()

            num_batches += 1

    return {
        "mean_iou": total_iou / num_batches,
        "dice_coef": total_dice / num_batches,
        "c_2": total_class_dice[2] / num_batches,
        "c_3": total_class_dice[3] / num_batches,
        "c_4": total_class_dice[4] / num_batches
    }


def magnitude_based_pruning(model, prune_ratio, val_loader=None):
    """
    Apply magnitude-based pruning to the model.

    The pruning process follows these steps:
    1. Calculate L2 norm magnitudes for all filters
    2. Determine threshold based on the pruning ratio
    3. Create binary masks to zero out weights below threshold
    4. Apply masks to model weights
    5. Calculate resulting sparsity

    Args:
        model: PyTorch model to prune
        prune_ratio: Percentage of weights to prune (0.0-1.0)
        val_loader: Optional validation loader for evaluation

    Returns:
        tuple: (pruned_model, sparsity_percentage)
    """
    # Calculate weight magnitudes
    magnitudes = calculate_weight_magnitudes(model)

    # Create pruning masks
    threshold_percentile = prune_ratio * 100
    masks = create_pruning_mask(magnitudes, threshold_percentile)

    # Apply pruning masks
    pruned_model = apply_pruning_masks(model, masks)

    # Calculate sparsity (percentage of zeroed weights)
    total_weights = 0
    zero_weights = 0
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()

    sparsity = 100.0 * zero_weights / total_weights

    return pruned_model, sparsity, masks


def main():
    """
    Main function to execute the pruning pipeline.

    The pipeline consists of:
    1. Loading the original model
    2. Evaluating the original model as baseline
    3. Applying pruning at different ratios
    4. Evaluating each pruned model
    5. Selecting the best model based on size-performance trade-off
    6. Saving all models and results
    """
    # Load the original model
    model_path = 'model/dlu_net_model_epoch_20.pth'
    original_model = load_trained_model(model_path)
    original_model.to(device)

    # Create a copy for pruning
    model = copy.deepcopy(original_model)

    # Print original model size
    original_size = get_model_size(original_model)
    print(f"Original Model size: {original_size / 1e6:.2f}M parameters")

    # Load validation data
    train_data = r"./data"
    val_dataset = BrainDataset(train_data, "val")
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Evaluate original model
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, val_loader)
    print(f"Original model metrics: {original_metrics}")

    # Define pruning ratios to try
    # Based on empirical studies, optimal pruning typically falls in the 20-60% range
    # for neural networks (Han et al., 2015; Zhu & Gupta, 2017)
    pruning_ratios = [0.3, 0.4, 0.5]

    results = []
    for ratio in pruning_ratios:
        print(f"\nApplying magnitude-based pruning with ratio {ratio}...")

        # Reset model to original weights
        model = copy.deepcopy(original_model)

        # Apply pruning
        pruned_model, sparsity, masks = magnitude_based_pruning(
            model, ratio, val_loader)
        # Compute channel pruning information
        channels_pruned = 0
        total_channels = 0
        for name, mask in masks.items():
            if mask.dim() == 1:
                total_channels += mask.shape[0]
                channels_pruned += (mask == 0).sum().item()

        # Check model size after pruning
        pruned_size = get_model_size(pruned_model)
        print(f"Pruned Model size: {pruned_size / 1e6:.2f}M parameters")
        print(
            f"Parameter reduction: {100 * (1 - pruned_size / original_size):.2f}%")
        print(f"Weight sparsity: {sparsity:.2f}%")

        # Evaluate pruned model
        print(f"Evaluating pruned model...")
        pruned_metrics = evaluate_model(pruned_model, val_loader)
        print(f"Pruned model metrics: {pruned_metrics}")

        # Save pruned model
        output_path = f'model/magnitude_pruned_model_{int(ratio*100)}.pth'
        torch.save(pruned_model, output_path)
        print(f"Saved pruned model to {output_path}")

        # Record results
        result = {
            "pruning_ratio": ratio,
            "original_size": original_size,
            "pruned_size": pruned_size,
            "reduction_percent": 100 * (1 - pruned_size / original_size),
            "weight_sparsity": sparsity,
            "original_metrics": original_metrics,
            "pruned_metrics": pruned_metrics,
            # Pruning details
            "pruning_type": "Magnitude",
            "original_params": int(original_size),
            "pruned_params": int(pruned_size),
            "model_size_before_mb": original_size/1e6,
            "model_size_after_mb": pruned_size/1e6,
            "sparsity": sparsity,
            "channels_pruned": channels_pruned,
            "total_channels": total_channels
        }
        results.append(result)

    # Save all results to file
    with open('model/magnitude_pruning_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nAll results saved to model/magnitude_pruning_results.json")

    # Find best model based on metric preservation and size reduction
    # This is a multi-objective optimization problem where we want to:
    # 1. Maximize performance (typically measured by Dice coefficient for segmentation)
    # 2. Maximize parameter reduction
    best_model_idx = 0
    best_score = 0
    for i, result in enumerate(results):
        # Calculate a combined score (size reduction * performance preservation)
        metric_preservation = result["pruned_metrics"]["dice_coef"] / \
            original_metrics["dice_coef"]
        size_reduction = result["reduction_percent"] / 100
        combined_score = metric_preservation * size_reduction

        if combined_score > best_score:
            best_score = combined_score
            best_model_idx = i

    best_result = results[best_model_idx]
    print(f"\nBest pruning configuration:")
    print(f"Pruning ratio: {best_result['pruning_ratio']}")
    print(f"Parameter reduction: {best_result['reduction_percent']:.2f}%")
    print(
        f"Dice coefficient: {best_result['pruned_metrics']['dice_coef']:.4f} (original: {original_metrics['dice_coef']:.4f})")
    print(f"Weight sparsity: {best_result['weight_sparsity']:.2f}%")




def statistics():
    device = torch.device("mps")

    data_path = "data/"
    val_dataset = BrainDataset(data_path, "val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    shared_model = load_trained_model("model/pruned_pretrained/magnitude/dlu_net_model_epoch_10.pth")
    original_model = load_trained_model("model/dlu_net_model_epoch_58.pth")

    shared_model = shared_model.to(device)
    original_model = original_model.to(device)


    X_val, y_val = next(iter(val_loader))
    X_val, y_val = X_val.to(device), y_val.to(device)


    # 5. compare the current base model vs this model
    print("\n===== PERFORMANCE METRICS =====")

    # Create and save comparison metrics
    original_model_metrics = ModelPerformanceMetrics("Original_DLU_Net")
    original_metrics = original_model_metrics.extract_metrics_from_model(
        original_model
    )

    pruned_model_metrics = ModelPerformanceMetrics("Magnitude Pruned Model")
    pruned_metrics = pruned_model_metrics.extract_metrics_from_model(
        shared_model
    )

    pruned_model_metrics.benchmark_inference_speed(shared_model, X_val)
    original_model_metrics.benchmark_inference_speed(original_model, X_val)


    model_comparison = ModelComparison(
        original_model_metrics, pruned_model_metrics)

    model_comparison.calculate_speedup()
    model_comparison.print_summary()




if __name__ == "__main__":
    # main()
    statistics()
