from src.utils.mlflow import setup_experiment
from src.architecture.model import load_trained_model, mean_iou, class_dice, get_initial_model
from utils.custom_metric import dice_coef
from utils.custom_loss import Weighted_BCEnDice_loss
from torch.utils.data import DataLoader
from src.utils.load_data import BrainDataset, get_data_loaders
from src.utils.stats import ModelPerformanceMetrics, ModelComparison
import torch
import torch.nn as nn
import numpy as np
import copy
import os
import json
import sys
import time
from pathlib import Path

# Set device
device = torch.device("cuda")


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
    loss = Weighted_BCEnDice_loss(masks, outputs)

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
    results = []

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            # Get the first sample
            sample_image = val_images.to(device)
            sample_mask = val_masks.to(device)

            # Get prediction
            prediction = model(sample_image)
            thresholded_pred = (prediction > 0.2).float()

            # Use the prediction directly for dice calculation instead of calling evaluate_dice_scores
            tc_dice = class_dice(thresholded_pred, sample_mask, 2).item()
            ec_dice = class_dice(thresholded_pred, sample_mask, 3).item()
            wt_dice = class_dice(thresholded_pred, sample_mask, 4).item()

            dice_score_main = dice_coef(thresholded_pred, sample_mask).item()
            mean_iou_score = mean_iou(thresholded_pred, sample_mask).item()

            results.append([tc_dice, ec_dice, wt_dice,
                            dice_score_main, mean_iou_score])
        print(
            f"Sample Dice Scores - Tumor Core: {tc_dice:.4f}, Enhancing Tumor: {ec_dice:.4f}, Whole Tumor: {wt_dice:.4f}")

    # get average of the results
    print(results)
    avg_tc_dice = sum([x[0] for x in results]) / len(results)
    avg_ec_dice = sum([x[1] for x in results]) / len(results)
    avg_wt_dice = sum([x[2] for x in results]) / len(results)
    avg_dice_score_main = sum([x[3] for x in results]) / len(results)
    avg_mean_iou = sum([x[4] for x in results]) / len(results)
    # Calculate mean metrics
    mean_metrics = {
        "mean_iou": avg_mean_iou,
        "dice_coef": avg_dice_score_main,
        "c_2": avg_tc_dice,
        "c_3": avg_ec_dice,
        "c_4": avg_wt_dice
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


def iterative_snip_pruning(model, iteration_prune_ratio, num_iterations, val_loader=None):
    """
    Apply iterative SNIP pruning to the model over multiple iterations.

    Args:
        model: PyTorch model to prune
        iteration_prune_ratio: Percentage of weights to prune per iteration (0.0-1.0)
        num_iterations: Number of pruning iterations
        val_loader: Optional validation loader for evaluation

    Returns:
        tuple: (pruned_model, sparsity_percentage)
    """
    start_time = time.time()
    current_model = copy.deepcopy(model)
    cumulative_sparsity = 0.0

    print(
        f"Starting iterative pruning ({num_iterations} iterations with {iteration_prune_ratio * 100:.1f}% per iteration)")

    for i in range(num_iterations):
        print(f"\nIteration {i + 1}/{num_iterations}")
        # Recalculate pruning ratio for this iteration
        # We need to adjust the pruning ratio to account for previously pruned weights
        if i > 0:
            # Calculate the adjusted pruning ratio for this iteration
            remaining_weights_ratio = 1.0 - (cumulative_sparsity / 100.0)
            # How many weights to prune from the remaining weights
            effective_prune_ratio = iteration_prune_ratio / remaining_weights_ratio
        else:
            effective_prune_ratio = iteration_prune_ratio

        current_model, sparsity, _ = snip_pruning(current_model, effective_prune_ratio, val_loader)

        # Calculate actual sparsity after this iteration
        total_weights = 0
        zero_weights = 0
        for name, param in current_model.named_parameters():
            if 'weight' in name and param.requires_grad:
                total_weights += param.numel()
                zero_weights += (param == 0).sum().item()

        cumulative_sparsity = 100.0 * zero_weights / total_weights
        print(f"  Iteration {i + 1} - Current sparsity: {cumulative_sparsity:.2f}%")

        # Optional: Evaluate model after each iteration
        if val_loader is not None:
            metrics = evaluate_model(current_model, val_loader)
            print(f"  Dice: {metrics['dice_coef']:.4f}, mIoU: {metrics['mean_iou']:.4f}")

    training_time = time.time() - start_time

    # Calculate channel pruning information
    channels_pruned = 0
    total_channels = 0
    for name, param in current_model.named_parameters():
        if 'weight' in name and param.requires_grad and param.dim() == 4:
            total_channels += param.shape[0]
            per_channel_sum = param.view(param.shape[0], -1).sum(dim=1)
            channels_pruned += (per_channel_sum == 0).sum().item()

    return current_model, cumulative_sparsity, training_time, channels_pruned, total_channels


def run_pruning_experiment(model_path, pruning_type='one-shot', prune_ratio=0.3,
                           iteration_ratio=0.05, num_iterations=6):

    # Load the original model
    original_model = load_trained_model(model_path)
    original_model.to(device)

    # load initial model
    model = get_initial_model()

    # Print original model size
    original_size = get_model_size(original_model)
    print(f"Original Model size: {original_size / 1e6:.2f}M parameters")

    # Prepare validation dataset
    train_loader, val_loader, test_loader = get_data_loaders("data")

    # Evaluate original model
    print("Evaluating original model...")
    original_metrics = evaluate_model(original_model, test_loader)
    print("Original model metrics:")
    for metric, value in original_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Apply pruning based on type
    start_time = time.time()

    if pruning_type == 'one-shot':
        print(f"\nApplying one-shot SNIP pruning (ratio: {prune_ratio * 100:.1f}%)...")
        pruned_model, sparsity, pruning_masks = snip_pruning(
            model, prune_ratio=prune_ratio, val_loader=val_loader)

        # Compute channel pruning information
        channels_pruned = 0
        total_channels = 0
        for name, mask in pruning_masks.items():
            if mask.dim() == 4:
                total_channels += mask.shape[0]
                per_channel_sum = mask.view(mask.shape[0], -1).sum(dim=1)
                channels_pruned += (per_channel_sum == 0).sum().item()

        training_time = time.time() - start_time

    elif pruning_type == 'iterative':
        print(
            f"\nApplying iterative SNIP pruning ({num_iterations} iterations, {iteration_ratio * 100:.1f}% per iteration)...")
        pruned_model, sparsity, training_time, channels_pruned, total_channels = iterative_snip_pruning(
            model, iteration_ratio, num_iterations, test_loader)

    else:
        raise ValueError("Pruning type must be 'one-shot' or 'iterative'")

    # Recalculate actual non-zero parameters
    total_weights = 0
    zero_weights = 0
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and param.requires_grad:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()

    nonzero_params = total_weights - zero_weights
    pruned_size = nonzero_params
    param_reduction = (original_size - pruned_size) * 100 / original_size

    print(f"Pruned Model size: {pruned_size / 1e6:.2f}M parameters")
    print(f"Sparsity achieved: {sparsity:.2f}%")
    print(f"Parameter reduction: {param_reduction:.2f}%")
    print(f"Training time: {training_time:.2f} seconds")

    # Evaluate pruned model
    print("\nEvaluating pruned model...")
    pruned_metrics = evaluate_model(pruned_model, val_loader)
    print("Pruned model metrics:")
    for metric, value in pruned_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save pruned model
    model_filename = f"snip_{pruning_type}_{'_'.join([str(int(prune_ratio * 100)) if pruning_type == 'one-shot' else str(int(iteration_ratio * 100)) + 'x' + str(num_iterations)])}.pth"
    model_save_path = os.path.join('model/snip/', model_filename)
    torch.save(pruned_model.state_dict(), model_save_path)
    print(f"Pruned model saved to {model_save_path}")

    # Save and compare on-disk sizes
    dense_mb = os.path.getsize(model_path) / (1024 * 1024)
    sparse_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"Dense saved model size: {dense_mb:.2f} MB")
    print(f"Sparse saved model size: {sparse_mb:.2f} MB")

    # Create metrics dictionary
    results = {
        "pruning_type": pruning_type,
        "prune_ratio": prune_ratio if pruning_type == 'one-shot' else iteration_ratio * num_iterations,
        "iteration_ratio": iteration_ratio if pruning_type == 'iterative' else None,
        "num_iterations": num_iterations if pruning_type == 'iterative' else None,
        "original_size": original_size,
        "pruned_size": pruned_size,
        "sparsity": sparsity,
        "parameter_reduction": param_reduction,
        "training_time": training_time,
        "original_metrics": original_metrics,
        "pruned_metrics": pruned_metrics,
        "channels_pruned": channels_pruned,
        "total_channels": total_channels,
        "model_path": model_save_path,
        "on_disk_size_mb": sparse_mb
    }

    return results


def save_comparison_table(results_list, output_path='model/snip/snip_comparison_table.json'):
    """
    Save the comparison table data to a JSON file.

    Args:
        results_list: List of result dictionaries from run_pruning_experiment
        output_path: Path to save the comparison table
    """
    table_data = {
        "comparison_table": [
            {
                "method": "Baseline",
                "pruning_ratio": "0%",
                "dice": results_list[0]["original_metrics"]["dice_coef"],
                "miou": results_list[0]["original_metrics"]["mean_iou"],
                "param_reduction": "0%",
                "training_time": "0"
            }
        ]
    }

    # Add each pruning method to the table
    for result in results_list:
        if result["pruning_type"] == "one-shot":
            method_name = f"One-Shot ({int(result['prune_ratio'] * 100)}%)"
        else:
            method_name = f"Iterative ({int(result['iteration_ratio'] * 100)}% × {result['num_iterations']})"

        table_data["comparison_table"].append({
            "method": method_name,
            "pruning_ratio": f"{result['prune_ratio'] * 100:.1f}%",
            "dice": result["pruned_metrics"]["dice_coef"],
            "miou": result["pruned_metrics"]["mean_iou"],
            "param_reduction": f"{result['parameter_reduction']:.2f}%",
            "training_time": f"{result['training_time']:.2f}s"
        })

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(table_data, f, indent=4)

    print(f"Comparison table saved to {output_path}")

    # Print table for easy copy-paste to LaTeX
    print("\nSNIP Pruning Comparison Table:")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{SNIP Pruning: One-Shot vs. Iterative Comparison}")
    print("\\label{tab:snip_comparison}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print(
        "\\textbf{Method} & \\textbf{Pruning Ratio} & \\textbf{Dice} & \\textbf{mIoU} & \\textbf{Param. Reduction} & \\textbf{Training Time} \\\\")
    print("\\midrule")

    for entry in table_data["comparison_table"]:
        dice_val = f"{entry['dice']:.4f}" if isinstance(entry['dice'], float) else entry['dice']
        miou_val = f"{entry['miou']:.4f}" if isinstance(entry['miou'], float) else entry['miou']
        print(
            f"{entry['method']} & {entry['pruning_ratio']} & {dice_val} & {miou_val} & {entry['param_reduction']} & {entry['training_time']} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")


def main():
    """
    Main function to execute multiple SNIP pruning experiments.
    """
    model_path = 'model/base_model/dlu_net_model_epoch_35.pth'
    results = []

    # Run baseline evaluation (already included in each experiment, but we'll use the first one's baseline)
    print("\n===== Running Experiment 1: One-Shot 10% Pruning =====")
    results.append(run_pruning_experiment(model_path, 'one-shot', prune_ratio=0.1))

    print("\n===== Running Experiment 2: One-Shot 20% Pruning =====")
    results.append(run_pruning_experiment(model_path, 'one-shot', prune_ratio=0.2))

    print("\n===== Running Experiment 2: One-Shot 30% Pruning =====")
    results.append(run_pruning_experiment(model_path, 'one-shot', prune_ratio=0.3))

    # print("\n===== Running Experiment 3: Iterative Pruning (10% × 6) =====")
    # results.append(run_pruning_experiment(model_path, 'iterative', iteration_ratio=0.10, num_iterations=6))
    #
    # print("\n===== Running Experiment 4: Iterative Pruning (20% × 3) =====")
    # results.append(run_pruning_experiment(model_path, 'iterative', iteration_ratio=0.20, num_iterations=3))

    # Save detailed results for each experiment
    for i, result in enumerate(results):
        output_path = f"model/snip/snip_experiment_{i + 1}_details.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Detailed results for experiment {i + 1} saved to {output_path}")

    # Generate the comparison table for LaTeX
    save_comparison_table(results)

    # Optionally log to MLflow
    for result in results:
        config = {
            "pruning_type": result["pruning_type"],
            "prune_ratio": result["prune_ratio"]
        }
        if result["pruning_type"] == "iterative":
            config.update({
                "iteration_ratio": result["iteration_ratio"],
                "num_iterations": result["num_iterations"]
            })

        experiment = setup_experiment(f"SNIP {result['pruning_type'].title()} Pruning", None, config)
        experiment.log_metrics({
            "dice_coef": result["pruned_metrics"]["dice_coef"],
            "mean_iou": result["pruned_metrics"]["mean_iou"],
            "sparsity": result["sparsity"],
            "parameter_reduction": result["parameter_reduction"],
            "training_time": result["training_time"]
        })
        experiment.log_artifact(result["model_path"])
        experiment.end_run()


if __name__ == "__main__":
    main()
