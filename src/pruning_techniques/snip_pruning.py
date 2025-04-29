from src.utils.mlflow import setup_experiment
from src.utils.custom_loss import weighted_bce_loss
from src.utils.custom_metric import dice_coef, mean_iou
from src.architecture.model import load_trained_model, class_dice
from torch.utils.data import DataLoader
from src.utils.load_data import BrainDataset
from src.utils.stats import ModelPerformanceMetrics, ModelComparison
import torch
import torch.nn as nn
import numpy as np
import copy
import os
import json
import sys
from pathlib import Path

# Import project modules after setting up sys.path


# Set device
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


def prepare_data_batch(batch_size=16):
    """
    Prepare a batch of data for computing connection sensitivity.

    Args:
        batch_size: Size of the batch to use

    Returns:
        tuple: (images, masks) training batch
    """
    train_data = r"./data"
    dataset = BrainDataset(train_data, "train")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, masks = next(iter(dataloader))
    return images.to(device), masks.to(device)


def calculate_connection_sensitivity(model, data_batch, prune_ratio):
    """
    Calculate the connection sensitivity for each weight using SNIP method.

    Args:
        model: PyTorch model
        data_batch: (images, masks) tuple for training
        prune_ratio: Percentage of weights to prune (0.0-1.0)

    Returns:
        tuple: (pruned_model, sensitivity_scores, pruning_mask)
    """
    # Create a copy of the model to compute sensitivity
    model.zero_grad()
    images, masks = data_batch

    # Register hooks for gradient computation
    sensitivity_scores = {}
    pruning_masks = {}

    # Get all trainable weights
    weights_to_prune = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            # Include all weights in pruning
            weights_to_prune.append((name, param))

    # Forward pass
    outputs = model(images)

    # Create a weight tensor for the loss function (all ones for simplicity)
    weight = torch.ones_like(masks)
    loss = weighted_bce_loss(masks, outputs, weight)

    # Backward pass to compute gradients
    loss.backward()

    # Calculate sensitivity scores
    all_scores = []
    for name, weight in weights_to_prune:
        # Calculate sensitivity score: |weight * gradient|
        # Keep on same device as the model
        score = torch.abs(weight.grad * weight).detach()
        sensitivity_scores[name] = score
        all_scores.append(score.view(-1))

    # Flatten all scores and determine global threshold using numpy
    # Convert to numpy for percentile calculation
    all_scores_tensor = torch.cat(all_scores)
    all_scores_numpy = all_scores_tensor.detach().cpu().numpy()

    # Calculate percentile threshold
    percentile_value = prune_ratio * 100
    threshold = float(np.percentile(all_scores_numpy, percentile_value))

    # Create masks based on threshold
    for name, score in sensitivity_scores.items():
        pruning_masks[name] = (score > threshold).float()

    # Apply masks to model
    pruned_model = copy.deepcopy(model)
    for name, param in pruned_model.named_parameters():
        if name in pruning_masks:
            # Ensure mask is on the same device as the parameter
            mask = pruning_masks[name].to(param.device)
            param.data = param.data * mask

    return pruned_model, sensitivity_scores, pruning_masks


def evaluate_model(model, val_loader):
    """
    Evaluate model on validation data.

    Args:
        model: PyTorch model
        val_loader: Validation data loader

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

    # Calculate mean metrics
    mean_metrics = {
        "mean_iou": total_iou / num_batches,
        "dice_coef": total_dice / num_batches,
        "c_2": total_class_dice[2] / num_batches,
        "c_3": total_class_dice[3] / num_batches,
        "c_4": total_class_dice[4] / num_batches
    }

    return mean_metrics


def snip_pruning(model, prune_ratio=0.3, val_loader=None):
    """
    Apply SNIP pruning to the model.

    The pruning process follows these steps:
    1. Get a batch of training data
    2. Calculate connection sensitivity for each weight
    3. Create binary masks based on sensitivity scores and pruning ratio
    4. Apply masks to model weights
    5. Calculate resulting sparsity

    Args:
        model: PyTorch model to prune
        prune_ratio: Percentage of weights to prune (0.0-1.0)
        val_loader: Optional validation loader for evaluation

    Returns:
        tuple: (pruned_model, sparsity_percentage)
    """
    # Get a batch of training data
    data_batch = prepare_data_batch()

    # Calculate connection sensitivity and create pruned model
    pruned_model, sensitivity_scores, pruning_masks = calculate_connection_sensitivity(
        model, data_batch, prune_ratio)

    # Calculate sparsity (percentage of zeroed weights)
    total_weights = 0
    zero_weights = 0
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and param.requires_grad:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()

    sparsity = 100.0 * zero_weights / total_weights

    return pruned_model, sparsity, pruning_masks


def main():
    """
    Main function to execute the SNIP pruning pipeline.
    """
    # Load the original model
    model_path = 'model/dlu_net_model_best.pth'
    original_model = load_trained_model(model_path)
    original_model.to(device)

    # Create a copy for pruning
    model = copy.deepcopy(original_model)

    # Print original model size
    original_size = get_model_size(original_model)
    print(f"Original Model size: {original_size / 1e6:.2f}M parameters")

    # Prepare validation dataset
    train_data = r"./data"
    val_dataset = BrainDataset(train_data, "val")
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Evaluate original model
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, val_loader)
    print("Original model metrics:")
    for metric, value in original_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Apply SNIP pruning
    print("\nApplying SNIP pruning...")
    pruned_model, sparsity, pruning_masks = snip_pruning(
        model, prune_ratio=0.3, val_loader=val_loader)
    # Compute channel pruning information
    channels_pruned = 0
    total_channels = 0
    for name, mask in pruning_masks.items():
        if mask.dim() == 4:
            total_channels += mask.shape[0]
            per_channel_sum = mask.view(mask.shape[0], -1).sum(dim=1)
            channels_pruned += (per_channel_sum == 0).sum().item()

    # Print pruning results: recalc actual non-zero parameters
    total_weights = 0
    zero_weights = 0
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and param.requires_grad:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()
    nonzero_params = total_weights - zero_weights
    pruned_size = nonzero_params
    print(f"Pruned Model size: {pruned_size / 1e6:.2f}M parameters")
    print(f"Sparsity achieved: {sparsity:.2f}%")
    print(
        f"Parameter reduction: {(original_size - pruned_size) * 100 / original_size:.2f}%")

    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_metrics = evaluate_model(pruned_model, val_loader)
    print("Pruned model metrics:")
    for metric, value in pruned_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save pruned model
    model_save_path = 'model/snip_pruned_model.pth'
    torch.save(pruned_model.state_dict(), model_save_path)
    print(f"Pruned model saved to {model_save_path}")
    # Save and compare on-disk sizes
    # use original model file for dense size
    dense_mb = os.path.getsize(model_path) / (1024 * 1024)
    sparse_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"Dense saved model size: {dense_mb:.2f} MB")
    print(f"Sparse saved model size: {sparse_mb:.2f} MB")

    # Create and save comparison metrics
    original_model_metrics = ModelPerformanceMetrics("Original_DLU_Net")
    original_model_metrics.record_all_metrics(
        original_metrics
    )

    pruned_model_metrics = ModelPerformanceMetrics("SNIP_Pruned_DLU_Net")
    pruned_model_metrics.record_all_metrics(
        pruned_metrics
    )
    pruned_model_metrics.record_sparsity_reduction(sparsity)

    # Compare models
    model_comparison = ModelComparison(
        original_model_metrics, pruned_model_metrics)
    model_comparison.print_summary()

    # Save metrics to file
    metrics_data = {
        "original": original_model_metrics.to_dict(),
        "pruned": pruned_model_metrics.to_dict(),
        "comparison_summary": model_comparison.compare_metrics()
    }
    # Record pruning details
    metrics_data["pruning_details"] = {
        "pruning_type": "SNIP",
        "original_params": int(original_size),
        "pruned_params": int(pruned_size),
        "model_size_before_mb": original_size/1e6,
        "model_size_after_mb": pruned_size/1e6,
        "sparsity": sparsity,
        "channels_pruned": channels_pruned,
        "total_channels": total_channels
    }

    with open('model/snip_pruning_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("\nMetrics saved to model/snip_pruning_metrics.json")

    # Log to MLflow
    config = {"prune_ratio": 0.3}
    experiment = setup_experiment(
        "SNIP Pruning Experiment", original_model, config)
    experiment.log_pruning_details(metrics_data["pruning_details"])
    # Log only numeric metrics dict to MLflow, filtering out None values
    orig_metrics = {k: v for k, v in metrics_data["original"]["metrics"].items(
    ) if isinstance(v, (int, float))}
    pruned_metrics = {k: v for k, v in metrics_data["pruned"]["metrics"].items(
    ) if isinstance(v, (int, float))}
    experiment.log_metrics(orig_metrics)
    experiment.log_metrics(pruned_metrics)
    # Flatten comparison summary for logging
    flat_comparison = {}
    for metric, comp in metrics_data["comparison_summary"].items():
        flat_comparison[f"{metric}_baseline"] = comp["baseline"]
        flat_comparison[f"{metric}_pruned"] = comp["pruned"]
        flat_comparison[f"{metric}_change"] = comp["change"]
        flat_comparison[f"{metric}_change_pct"] = comp["change_pct"]
    experiment.log_metrics(flat_comparison)
    experiment.log_state_dict(model_save_path)
    experiment.end_run()


if __name__ == "__main__":
    main()
