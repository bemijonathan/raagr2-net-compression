"""
SNIP Pruning for Neural Networks

This script implements SNIP (Single-shot Network Pruning based on Connection Sensitivity)
for neural network compression, based on the approach described in:

"SNIP: Single-shot Network Pruning based on Connection Sensitivity"
by Namhoon Lee, Thalaiyasingam Ajanthan, Philip H. S. Torr (2019)
https://arxiv.org/abs/1810.02340

SNIP prunes networks at initialization based on a saliency criterion that identifies
structurally important connections for the given task. This approach:
1. Requires only a single computation of connection sensitivities using a small batch of data
2. Prunes the network once at initialization before training
3. Eliminates the need for both pretraining and complex pruning schedules
"""

import torch
import torch.nn as nn
from utils.stats import ModelPerformanceMetrics, ModelComparison
from utils.load_data import get_data_loaders
from architecture.model import load_trained_model,get_initial_model, class_dice, dice_coef, mean_iou
import json
import copy
import numpy as np

# Set device to Metal Performance Shaders (MPS) for accelerated computation on Mac
device = torch.device("mps")


train_loader, val_loader, test_loader = get_data_loaders("data")



def get_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_connection_sensitivity(model, dataloader, criterion):
    """
    Calculate the connection sensitivity for each parameter in the model.

    Following the SNIP methodology, we compute the sensitivity as the absolute
    value of the product of the parameter and its gradient after a single
    forward-backward pass through a batch of data.

    Args:
        model: PyTorch model at initialization
        dataloader: DataLoader containing a single batch of data
        criterion: Loss function to use for the backward pass

    Returns:
        dict: Dictionary mapping parameter names to their sensitivity tensors
    """
    # Enable gradient computation for all parameters
    for param in model.parameters():
        if param.requires_grad:
            # this means if the parameter is trainable
            param.requires_grad_(True)

    # Get a single batch of data
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)

    # get the predicted masks
    outputs = model(images)
    loss = criterion(outputs, masks)

    # Backward pass to compute gradients
    loss.backward()

    # Calculate sensitivity scores
    sensitivities = {}
    for name, param in model.named_parameters():
        # if weight and grad is not None
        if 'weight' in name and param.grad is not None:
            # SNIP sensitivity score: |w * grad_w|
            sensitivities[name] = torch.abs(param.grad * param)

    # Reset gradients
    model.zero_grad()

    return sensitivities


def create_pruning_mask(sensitivities, prune_ratio):
    """
    Create pruning masks based on connection sensitivities.

    We keep the connections with the highest sensitivity scores, pruning
    the specified percentage of connections with the lowest scores.

    Args:
        sensitivities: Dictionary mapping parameter names to sensitivity tensors
        prune_ratio: Percentage of weights to prune (0.0-1.0)

    Returns:
        dict: Dictionary of binary masks for each parameter
    """
    masks = {}

    # Flatten all sensitivity scores into a single tensor for global pruning
    all_scores = torch.cat([s.view(-1) for s in sensitivities.values()])

    # Compute threshold using topk instead of kthvalue for MPS compatibility
    if prune_ratio >= 1.0:
        threshold = float('inf')
    else:
        # Calculate how many parameters to keep
        keep_ratio = 1.0 - prune_ratio
        num_params_to_keep = int(all_scores.numel() * keep_ratio)
        # Get the top-k values (k = number to keep)
        if num_params_to_keep > 0:
            # Move to CPU for compatibility if needed
            topk_values = torch.topk(all_scores, num_params_to_keep, sorted=True).values
            # The threshold is the smallest value among the top-k
            threshold = topk_values[-1].item()
        else:
            # No parameters to keep, set a very high threshold value
            threshold = float('inf')  # Direct float value doesn't have .item() method

    # Create masks: keep weights with sensitivity > threshold
    for name, sensitivity in sensitivities.items():
        masks[name] = (sensitivity > threshold).float()

    return masks


def apply_pruning_masks(model, masks):
    """
    Apply pruning masks to model parameters.

    This function zeroes out weights according to the pruning masks.

    Args:
        model: PyTorch model
        masks: Dictionary of binary masks for each parameter

    Returns:
        model: Pruned PyTorch model
    """
    for name, param in model.named_parameters():
        if name in masks:
            # Apply mask to zero out pruned weights
            param.data.mul_(masks[name])

    return model


def evaluate_model(model, val_loader):
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
        "mean_iou": round(total_iou / num_batches, 2),
        "dice_coef": round(total_dice / num_batches, 2),
        "c_2": round(total_class_dice[2] / num_batches, 2),
        "c_3": round(total_class_dice[3] / num_batches, 2),
        "c_4": round(total_class_dice[4] / num_batches, 2)
    }


def snip_pruning(model, prune_ratio, train_loader, criterion=None):
    """
    Apply SNIP pruning to the model.

    The pruning process follows these steps:
    1. Calculate connection sensitivities using a single batch
    2. Create binary masks to keep most important connections
    3. Apply masks to model weights
    4. Calculate resulting sparsity

    Args:
        model: PyTorch model to prune
        prune_ratio: Percentage of weights to prune (0.0-1.0)
        train_loader: DataLoader containing training data
        criterion: Loss function to use (defaults to weighted BCE-Dice)

    Returns:
        tuple: (pruned_model, sparsity_percentage, masks)
    """
    # Default to weighted BCE-Dice loss if none specified
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    # Calculate connection sensitivities
    sensitivities = calculate_connection_sensitivity(model, train_loader, criterion)

    # Create pruning masks
    masks = create_pruning_mask(sensitivities, prune_ratio)

    # Apply pruning masks
    pruned_model = apply_pruning_masks(model, masks)

    # Calculate sparsity (percentage of zeroed weights)
    total_weights = 0
    zero_weights = 0
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()

    sparsity = 100.0 * zero_weights / total_weights

    return pruned_model, sparsity, masks


def main():
    """
    Main function to execute the SNIP pruning pipeline.

    The pipeline consists of:
    1. Loading the original model (at initialization or pretrained)
    2. Creating a small batch loader for sensitivity calculation
    3. Applying SNIP pruning at different ratios
    4. Training and evaluating each pruned model
    5. Saving all models and results
    """
    # Load the original model
    model_path = 'model/base_model/dlu_net_model_epoch_35.pth'
    original_model = load_trained_model(model_path)
    original_model.to(device)

    # Define pruning ratios to try (10%, 20%, 30% as requested)
    pruning_ratios = [0.1, 0.2, 0.3]

    # Define loss criterion for sensitivity calculation
    criterion = nn.BCEWithLogitsLoss()

    # Evaluate original model first
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, test_loader)
    print(f"Original model metrics: {original_metrics}")

    results = []
    for ratio in pruning_ratios:
        print(f"\nApplying SNIP pruning with ratio {ratio}...")

        # Reset model to original weights
        model = get_initial_model()

        # Apply SNIP pruning
        pruned_model, sparsity, masks = snip_pruning(model, ratio, train_loader, criterion)

        # Compute channel pruning information
        channels_pruned = 0
        total_channels = 0
        for name, mask in masks.items():
            if len(mask.shape) >= 2:  # For conv layers
                out_channels = mask.shape[0]
                total_channels += out_channels
                # Count fully pruned output channels
                pruned_channels = sum([mask[i].sum() == 0 for i in range(out_channels)])
                channels_pruned += pruned_channels

        # Check model size after pruning
        pruned_size = get_model_size(pruned_model)
        original_size = get_model_size(original_model)
        reduction_percent = 100 * (1 - pruned_size / original_size)

        print(f"Pruned Model size: {pruned_size / 1e6:.2f}M parameters")
        print(f"Size reduction: {reduction_percent:.2f}%")
        print(f"Weight sparsity: {sparsity:.2f}%")

        # Evaluate pruned model
        print("Evaluating pruned model...")
        pruned_metrics = evaluate_model(pruned_model, test_loader)
        print(f"Pruned model metrics: {pruned_metrics}")

        # Save pruned model
        output_path = f'model/snip/snip_pruned_model_{int(ratio*100)}.pth'
        torch.save(pruned_model, output_path)
        print(f"Saved pruned model to {output_path}")

        # Record results
        result = {
            "pruning_ratio": ratio,
            "pruned_size": pruned_size,
            "weight_sparsity": sparsity,
            "pruned_metrics": pruned_metrics,
            "reduction_percent": reduction_percent,
            # Pruning details
            "pruning_type": "SNIP",
            "pruned_params": int(pruned_size),
            "model_size_after_mb": pruned_size/1e6,
            "sparsity": sparsity,
            "channels_pruned": channels_pruned,
            "total_channels": total_channels
        }
        results.append(result)

    # Convert tensor values to Python native types for JSON serialization
    def convert_tensors_to_python(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors_to_python(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    # Convert results to JSON-serializable format
    json_serializable_results = convert_tensors_to_python(results)

    # Save all results to file
    with open('model/snip/snip_pruning_results.json', 'w') as f:
        json.dump(json_serializable_results, f, indent=4)
    print("\nAll results saved to model/snip/snip_pruning_results.json")

    # Find best model based on metric preservation and size reduction
    best_model_idx = 0
    best_score = 0
    for i, result in enumerate(results):
        # Calculate a combined score (size reduction * performance preservation)
        metric_preservation = result["pruned_metrics"]["dice_coef"] / original_metrics["dice_coef"]
        size_reduction = result["reduction_percent"] / 100
        combined_score = metric_preservation * size_reduction

        if combined_score > best_score:
            best_score = combined_score
            best_model_idx = i

    best_result = results[best_model_idx]
    print("\nBest pruning configuration:")
    print(f"Pruning ratio: {best_result['pruning_ratio']}")
    print(f"Parameter reduction: {best_result['reduction_percent']:.2f}%")
    print(f"Dice coefficient: {best_result['pruned_metrics']['dice_coef']:.4f}")
    print(f"Weight sparsity: {best_result['weight_sparsity']:.2f}%")


def statistics():
    device = torch.device("mps")

    snip_model = load_trained_model("model/snip/snip_pruned_model_20.pth")
    original_model = load_trained_model("model/base_model/dlu_net_model_epoch_35.pth")

    snip_model = snip_model.to(device)
    original_model = original_model.to(device)

    # 5. compare the current base model vs this model
    print("\n===== PERFORMANCE METRICS =====")

    # Create and save comparison metrics
    original_model_metrics = ModelPerformanceMetrics("Original_DLU_Net")
    original_model_metrics.extract_metrics_from_model(
        original_model
    )

    pruned_model_metrics = ModelPerformanceMetrics("SNIP Pruned Model")
    pruned_model_metrics.extract_metrics_from_model(
        snip_model
    )

    # Benchmark using entire validation loader
    pruned_model_metrics.benchmark_inference_speed(snip_model, val_loader)
    original_model_metrics.benchmark_inference_speed(
        original_model, val_loader)

    model_comparison = ModelComparison(
        original_model_metrics, pruned_model_metrics)

    model_comparison.calculate_speedup()
    model_comparison.print_summary()


if __name__ == "__main__":
    main()
    # statistics()
