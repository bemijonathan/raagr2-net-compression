"""
Network Slimming Implementation

This script implements network slimming for neural network compression,
based on the approach described in the paper:

"Learning Efficient Convolutional Networks through Network Slimming"
by Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang

Network slimming is a structured pruning technique that:
1. Introduces scaling factors after each channel in convolutional layers (using BN layers)
2. Applies L1 regularization to these scaling factors during training
3. Prunes channels with small scaling factors after training
4. Fine-tunes the pruned network for best performance

Key advantages:
- Reduces model size
- Decreases run-time memory footprint
- Lowers computational cost
- Preserves or even improves accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import copy
import json
import time

from architecture.model import DLUNet, weighted_bce_dice_loss, dice_coef, class_dice, load_trained_model
from utils.load_data import get_data_loaders
from utils.stats import ModelPerformanceMetrics
from src.CONFIGS import batch_size

# Set device
device = torch.device("cuda")


def get_bn_layers(model):
    """
    Get all batch normalization layers in the model.

    Args:
        model: PyTorch model

    Returns:
        list: List of batch normalization layers
    """
    bn_layers = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append(module)
    return bn_layers


def compute_bn_scale_factors(model):
    """
    Compute the scale factors (gamma values) of all BatchNorm layers.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary mapping layer names to their scale factors
    """
    scale_factors = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Each BatchNorm2d has a 'weight' parameter which corresponds to gamma (scaling factors)
            scale_factors[name] = module.weight.data.abs().cpu().numpy()
    return scale_factors


def train_with_channel_regularization(model, train_loader, val_loader,
                                      num_epochs=30, learning_rate=1e-4,
                                      weight_decay=1e-4, l1_lambda=1e-5):
    """
    Train the model with L1 regularization on BN scaling factors.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        l1_lambda: Strength of the L1 regularization on BN scaling factors

    Returns:
        model: Trained model
    """
    print(
        f"Starting training with channel regularization (l1_lambda={l1_lambda})...")

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, min_lr=0.00001, verbose=True
    )

    bn_layers = get_bn_layers(model)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Ensure masks are in [0, 1] range to prevent BCE loss errors
            if masks.min() < 0 or masks.max() > 1:
                print(f"Warning: Found mask values outside [0,1] range: min={masks.min().item()}, max={masks.max().item()}")
                masks = torch.clamp(masks, 0, 1)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss (weighted BCE and Dice loss)
            task_loss = weighted_bce_dice_loss(outputs, masks)

            # Add L1 regularization on BatchNorm scaling factors
            l1_reg = 0
            for bn_layer in bn_layers:
                l1_reg += torch.norm(bn_layer.weight, 1)

            # Total loss with L1 regularization
            loss = task_loss + l1_lambda * l1_reg

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            train_dice += dice_coef(outputs, masks).item()

        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = weighted_bce_dice_loss(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coef(outputs, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create directory if it doesn't exist
            os.makedirs('model/slimming', exist_ok=True)
            torch.save(model.state_dict(),
                       'model/slimming/model_with_sparsity.pth')
            print(f"Saved new best model with val_loss: {val_loss:.4f}")

        # Log progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s, "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    # Load the best model weights
    model.load_state_dict(torch.load('model/slimming/model_with_sparsity.pth'))
    return model


def prune_channels(model, prune_ratio):
    """
    Prune channels based on the scale factors in BN layers.

    Args:
        model: PyTorch model with BN scale factors
        prune_ratio: Percentage of channels to prune (0-100)

    Returns:
        pruned_model: Model with channels pruned
        channel_masks: Dictionary of binary masks for each BN layer
    """
    print(f"Pruning {prune_ratio}% of channels...")

    # Make a deep copy of the model to prune
    pruned_model = copy.deepcopy(model)

    # Compute scale factors
    scale_factors = {}
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            scale_factors[name] = module.weight.data.abs().cpu().numpy()

    # Flatten all scale factors to determine global threshold
    all_factors = np.concatenate(
        [factors for factors in scale_factors.values()])
    threshold = np.percentile(all_factors, prune_ratio)

    print(f"Pruning threshold: {threshold}")

    # Create channel masks based on threshold
    channel_masks = {}
    for name, factors in scale_factors.items():
        # Threshold each channel: 1 means keep, 0 means prune
        channel_masks[name] = (factors > threshold).astype(np.float32)
        print(
            f"Layer {name}: keeping {np.sum(factors > threshold)}/{len(factors)} channels")

    # Apply masks to BatchNorm layers
    for name, module in pruned_model.named_modules():
        if name in channel_masks and isinstance(module, nn.BatchNorm2d):
            # Create a mask tensor
            mask = torch.from_numpy(channel_masks[name]).float().to(device)

            # Apply mask to weight and bias
            module.weight.data.mul_(mask)
            if module.bias is not None:
                module.bias.data.mul_(mask)

    # Save the pruned model
    os.makedirs('model/slimming', exist_ok=True)
    torch.save(pruned_model.state_dict(),
               f'model/slimming/slimmed_model_{prune_ratio}.pth')

    # Also save the channel masks for reference
    with open(f'model/slimming/channel_masks_{prune_ratio}.json', 'w') as f:
        serializable_masks = {k: v.tolist() for k, v in channel_masks.items()}
        json.dump(serializable_masks, f)

    return pruned_model, channel_masks


def fine_tune_pruned_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    """
    Fine-tune the pruned model to recover accuracy.

    Args:
        model: Pruned PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for optimizer

    Returns:
        model: Fine-tuned model
    """
    print("Fine-tuning the pruned model...")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=3, min_lr=0.000001, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = weighted_bce_dice_loss(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            train_dice += dice_coef(outputs, masks).item()

        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = weighted_bce_dice_loss(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coef(outputs, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       f'model/slimming/fine_tuned_model.pth')
            print(
                f"Saved new best fine-tuned model with val_loss: {val_loss:.4f}")

        # Log progress
        epoch_time = time.time() - start_time
        print(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s, "
              f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

    # Load the best model weights
    model.load_state_dict(torch.load(f'model/slimming/fine_tuned_model.pth'))
    return model


def count_effective_parameters(model, channel_masks=None):
    """
    Count the effective number of parameters in the model.
    If channel_masks is provided, it will count only the non-pruned parameters.

    Args:
        model: PyTorch model
        channel_masks: Dictionary of binary masks for each BN layer

    Returns:
        int: Number of effective parameters
    """
    if channel_masks is None:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count effective parameters when masks are applied
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in channel_masks:
            # Count active channels
            mask = channel_masks[name]
            total_params += np.sum(mask) * 2  # weight and bias

        # TODO: Counting conv layer parameters more accurately would require
        # tracking which channels are active in both input and output
        # This is a simplification

    return total_params


def evaluate_model(model, val_loader, name=""):
    """
    Evaluate the model on validation data

    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        name: Name identifier for the model

    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    tc_dice = 0.0
    ec_dice = 0.0
    wt_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Overall metrics
            loss = weighted_bce_dice_loss(outputs, masks)
            dice = dice_coef(outputs, masks)

            # Class-specific metrics
            tc = class_dice(outputs, masks, 2)
            ec = class_dice(outputs, masks, 3)
            wt = class_dice(outputs, masks, 4)

            val_loss += loss.item()
            val_dice += dice.item()
            tc_dice += tc.item()
            ec_dice += ec.item()
            wt_dice += wt.item()

    # Calculate averages
    num_batches = len(val_loader)
    metrics = {
        "model_name": name,
        "loss": val_loss / num_batches,
        "dice": val_dice / num_batches,
        "tc_dice": tc_dice / num_batches,
        "ec_dice": ec_dice / num_batches,
        "wt_dice": wt_dice / num_batches
    }

    print(f"Evaluation of {name}:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Dice: {metrics['dice']:.4f}")
    print(f"  TC Dice: {metrics['tc_dice']:.4f}")
    print(f"  EC Dice: {metrics['ec_dice']:.4f}")
    print(f"  WT Dice: {metrics['wt_dice']:.4f}")

    return metrics


def slimming_pipeline(model_path, prune_ratio=30, train_epochs=20, fine_tune_epochs=10, l1_lambda=1e-5):
    """
    Complete pipeline for network slimming:
    1. Load pre-trained model
    2. Train with L1 regularization on BN scale factors
    3. Prune channels with small scale factors
    4. Fine-tune the pruned model
    5. Evaluate and compare models

    Args:
        model_path: Path to the pre-trained model
        prune_ratio: Percentage of channels to prune (0-100)
        train_epochs: Number of epochs for sparsity training
        fine_tune_epochs: Number of epochs for fine-tuning
        l1_lambda: Strength of L1 regularization

    Returns:
        dict: Results of the slimming pipeline
    """
    print("======= Starting Network Slimming Pipeline =======")

    # 1. Load data
    train_loader, val_loader, test_loader = get_data_loaders("data")
    # 2. Load the pre-trained model
    print(f"Loading pre-trained model from {model_path}...")
    original_model = load_trained_model(model_path)
    original_model.to(device)

    # 3. Evaluate the original model
    print("Evaluating original model...")
    original_metrics = evaluate_model(
        original_model, test_loader, "Original Model")
    original_params = sum(p.numel()
                          for p in original_model.parameters() if p.requires_grad)
    print(f"Original model parameters: {original_params}")

    # 4. Train with sparsity-inducing regularization
    print(f"Training with L1 regularization (lambda={l1_lambda})...")
    sparsity_model = copy.deepcopy(original_model)
    sparsity_model = train_with_channel_regularization(
        sparsity_model, train_loader, val_loader,
        num_epochs=train_epochs, l1_lambda=l1_lambda
    )

    # 5. Evaluate the sparsity-trained model
    print("Evaluating sparsity-trained model...")
    sparsity_metrics = evaluate_model(
        sparsity_model, test_loader, "Sparsity-Trained Model")

    # 6. Visualize the distribution of scaling factors
    scale_factors = compute_bn_scale_factors(sparsity_model)

    # 7. Prune channels based on scaling factors
    print(f"Pruning {prune_ratio}% of channels...")
    pruned_model, channel_masks = prune_channels(sparsity_model, prune_ratio)

    # 8. Evaluate the pruned model
    print("Evaluating pruned model (before fine-tuning)...")
    pruned_metrics = evaluate_model(
        pruned_model, test_loader, f"Pruned Model ({prune_ratio}%)")
    pruned_params = sum(p.numel()
                        for p in pruned_model.parameters() if p.requires_grad)
    print(f"Pruned model parameters: {pruned_params}")

    # 9. Fine-tune the pruned model
    print("Fine-tuning pruned model...")
    # fine_tuned_model = fine_tune_pruned_model(
    #     pruned_model, train_loader, val_loader,
    #     num_epochs=fine_tune_epochs
    # )

    # 10. Evaluate the fine-tuned model
    # print("Evaluating fine-tuned model...")
    # fine_tuned_metrics = evaluate_model(
    #     fine_tuned_model, val_loader, f"Fine-tuned Model ({prune_ratio}%)"
    # )

    # 11. Save results
    results = {
        "original_model": {
            "metrics": original_metrics,
            "parameters": original_params
        },
        "sparsity_model": {
            "metrics": sparsity_metrics,
            "parameters": original_params  # Same structure as original
        },
        "pruned_model": {
            "metrics": pruned_metrics,
            "parameters": pruned_params,
            "prune_ratio": prune_ratio
        }
        # "fine_tuned_model": {
        #     "metrics": fine_tuned_metrics,
        #     "parameters": pruned_params  # Same structure as pruned
        # }
    }

    # Save results to JSON
    os.makedirs('model/slimming', exist_ok=True)
    with open(f'model/slimming/slimming_results_{prune_ratio}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("======= Network Slimming Pipeline Completed =======")

    return results


def multi_pass_slimming(model_path, prune_ratios=[10, 20, 30],
                        train_epochs=15, fine_tune_epochs=10, l1_lambda=1e-5):
    """
    Apply the slimming pipeline in multiple passes with increasing prune ratios.

    Args:
        model_path: Path to the pre-trained model
        prune_ratios: List of prune ratios to apply in sequence
        train_epochs: Number of epochs for each sparsity training
        fine_tune_epochs: Number of epochs for each fine-tuning
        l1_lambda: Strength of L1 regularization

    Returns:
        dict: Results of multi-pass slimming
    """
    print("======= Starting Multi-Pass Network Slimming =======")

    current_model_path = model_path
    all_results = {}

    for i, prune_ratio in enumerate(prune_ratios):
        print(f"\n=== Pass {i+1}: Pruning {prune_ratio}% ===")

        # Apply slimming
        results = slimming_pipeline(
            current_model_path,
            prune_ratio=prune_ratio,
            train_epochs=train_epochs,
            fine_tune_epochs=fine_tune_epochs,
            l1_lambda=l1_lambda
        )

        # Store results
        all_results[f"pass_{i+1}_{prune_ratio}%"] = results

        # Update model path for next iteration
        current_model_path = f'model/slimming/fine_tuned_model.pth'

    # Save all results
    with open('model/slimming/multi_pass_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("======= Multi-Pass Network Slimming Completed =======")

    return all_results


if __name__ == "__main__":
    # Example usage
    model_path = 'model/base_model/dlu_net_model_epoch_35.pth'

    # Single pass slimming with different prune ratios
    slimming_pipeline(model_path, prune_ratio=10,
                      train_epochs=5, fine_tune_epochs=0)
    slimming_pipeline(model_path, prune_ratio=20,
                      train_epochs=5, fine_tune_epochs=0)
    slimming_pipeline(model_path, prune_ratio=30,
                      train_epochs=5, fine_tune_epochs=0)

    # Alternatively, use multi-pass slimming for more aggressive pruning
    # multi_pass_slimming(model_path, prune_ratios=[10, 20, 30], train_epochs=15, fine_tune_epochs=10)
