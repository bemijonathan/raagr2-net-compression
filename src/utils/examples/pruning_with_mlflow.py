import torch
import os
import sys
import importlib
from torch.utils.data import DataLoader
from architecture.model import DLUNet, load_trained_model
from utils.custom_loss import Weighted_BCEnDice_loss
from utils.custom_metric import dice_coef
from utils.load_data import BrainDataset
from utils.mlflow import MLFlowExperiment, setup_experiment, log_epoch_metrics, log_best_model
from typing import Dict, Any, Optional, List, Callable


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    sparsity: float = 0.0
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.

    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        sparsity: Current model sparsity level

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = Weighted_BCEnDice_loss(outputs, masks)
            dice = dice_coef(outputs, masks)

            val_loss += loss.item()
            val_dice += dice.item()

    # Calculate average metrics
    val_loss /= len(data_loader)
    val_dice /= len(data_loader)

    return {
        'loss': val_loss,
        'dice': val_dice,
        'sparsity': sparsity
    }


def prune_model_with_mlflow(
    model_path: str,
    pruning_method: str,
    config: Dict[str, Any],
    experiment_name: str = "brain_segmentation_pruning",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> None:
    """
    Prune a trained model with MLflow tracking.

    Args:
        model_path: Path to the trained model
        pruning_method: Pruning method to use ('magnitude' or 'degraph')
        config: Dictionary containing pruning configuration
        experiment_name: Name of the MLflow experiment
        run_name: Optional name for this specific run
        tracking_uri: Optional MLflow tracking server URI
    """
    # Extract configuration values
    data_path = config.get("data_path", "./data")
    batch_size = config.get("batch_size", 16)
    device_name = config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu")
    in_channels = config.get("in_channels", 4)
    out_channels = config.get("out_channels", 5)

    # Set up device
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs('model/pruned', exist_ok=True)

    # Load the trained model
    model = load_trained_model(model_path)
    model.to(device)

    # Initialize validation dataset and dataloader
    val_dataset = BrainDataset(data_path, "val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up MLflow experiment
    mlflow_experiment = setup_experiment(
        experiment_name=f"{experiment_name}_{pruning_method}",
        model=model,
        config=config,
        run_name=run_name,
        tracking_uri=tracking_uri
    )

    # Log the original model metrics as baseline
    baseline_metrics = evaluate_model(model, val_loader, device)
    mlflow_experiment.log_metrics({
        'baseline_loss': baseline_metrics['loss'],
        'baseline_dice': baseline_metrics['dice'],
        'baseline_sparsity': 0.0
    })

    # Import the appropriate pruning module based on pruning_method
    pruning_module = None
    if pruning_method == 'magnitude':
        import magnitude_based_pruning as pruning_module
    elif pruning_method == 'degraph':
        import degraph_pruning as pruning_module
    else:
        raise ValueError(f"Unsupported pruning method: {pruning_method}")

    # Get the pruning function from the module
    prune_function = getattr(pruning_module, 'prune_model', None)
    if prune_function is None:
        raise AttributeError(
            f"Could not find prune_model function in {pruning_method}")

    # Perform pruning at different sparsity levels
    sparsity_levels = config.get('sparsity_levels', [0.2, 0.4, 0.6, 0.8])

    for sparsity in sparsity_levels:
        print(f"Pruning with {pruning_method} to sparsity level {sparsity}")

        # Create a copy of the original model for this pruning level
        pruned_model = load_trained_model(model_path)
        pruned_model.to(device)

        # Apply pruning
        pruned_model = prune_function(pruned_model, sparsity)

        # Evaluate pruned model
        pruned_metrics = evaluate_model(
            pruned_model, val_loader, device, sparsity)

        # Log metrics for this pruning level
        metrics_prefix = f"sparsity_{int(sparsity*100)}"
        mlflow_experiment.log_metrics({
            f"{metrics_prefix}_loss": pruned_metrics['loss'],
            f"{metrics_prefix}_dice": pruned_metrics['dice'],
            f"{metrics_prefix}_sparsity": sparsity
        })

        # Save pruned model checkpoint
        pruned_model_path = f"model/pruned/dlu_net_model_{pruning_method}_sparsity_{int(sparsity*100)}.pth"
        torch.save(pruned_model.state_dict(), pruned_model_path)
        mlflow_experiment.log_state_dict(pruned_model_path)

        # Log model with specific sparsity level
        # This is helpful to compare models in the MLflow UI
        mlflow_experiment.model = pruned_model
        mlflow_experiment.log_model(
            artifact_path=f"model_sparsity_{int(sparsity*100)}",
            registered_model_name=f"brain_segmentation_dlunet_{pruning_method}_sparsity_{int(sparsity*100)}"
        )

        print(
            f"Pruned model (sparsity {sparsity}) - Loss: {pruned_metrics['loss']:.4f}, Dice: {pruned_metrics['dice']:.4f}")
        print('-' * 60)

    # Create a comparative visualization
    visualize_pruning_results(
        sparsity_levels=sparsity_levels,
        baseline_metrics=baseline_metrics,
        pruned_metrics=[
            evaluate_model(
                load_trained_model(
                    f"model/pruned/dlu_net_model_{pruning_method}_sparsity_{int(s*100)}.pth"),
                val_loader,
                device,
                s
            ) for s in sparsity_levels
        ],
        mlflow_experiment=mlflow_experiment,
        pruning_method=pruning_method
    )

    # End MLflow run
    mlflow_experiment.end_run()
    print(f"Pruning completed using {pruning_method} method.")


def visualize_pruning_results(
    sparsity_levels: List[float],
    baseline_metrics: Dict[str, float],
    pruned_metrics: List[Dict[str, float]],
    mlflow_experiment: MLFlowExperiment,
    pruning_method: str
) -> None:
    """
    Create and log visualizations of pruning results.

    Args:
        sparsity_levels: List of sparsity levels
        baseline_metrics: Metrics of the baseline model
        pruned_metrics: List of metrics for each pruning level
        mlflow_experiment: MLflow experiment instance
        pruning_method: Name of the pruning method used
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare data for plotting
    x_values = [0] + [s for s in sparsity_levels]
    y_loss = [baseline_metrics['loss']] + [m['loss'] for m in pruned_metrics]
    y_dice = [baseline_metrics['dice']] + [m['dice'] for m in pruned_metrics]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot loss vs sparsity
    ax1.plot(x_values, y_loss, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Sparsity Level', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss vs Sparsity ({pruning_method})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(x_values)
    ax1.set_xticklabels([f"{int(s*100)}%" for s in x_values])

    # Plot dice vs sparsity
    ax2.plot(x_values, y_dice, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Sparsity Level', fontsize=12)
    ax2.set_ylabel('Dice Coefficient', fontsize=12)
    ax2.set_title(
        f'Dice Coefficient vs Sparsity ({pruning_method})', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(x_values)
    ax2.set_xticklabels([f"{int(s*100)}%" for s in x_values])

    # Add horizontal line for baseline
    ax1.axhline(y=baseline_metrics['loss'], color='r',
                linestyle='--', alpha=0.5, label='Baseline')
    ax2.axhline(y=baseline_metrics['dice'], color='r',
                linestyle='--', alpha=0.5, label='Baseline')

    # Add legends
    ax1.legend()
    ax2.legend()

    plt.tight_layout()

    # Log the figure to MLflow
    mlflow_experiment.log_figure(fig, f"{pruning_method}_pruning_results.png")
    plt.close(fig)


if __name__ == "__main__":
    # Example configuration for magnitude-based pruning
    magnitude_config = {
        "data_path": "./data",
        "batch_size": 16,
        "in_channels": 4,
        "out_channels": 5,
        "device": "mps",  # Use "cuda" for NVIDIA GPUs, "mps" for Apple Silicon, or "cpu"
        "sparsity_levels": [0.2, 0.4, 0.6, 0.8],
        "pruning_method": "magnitude",
        "global_pruning": True,
        "notes": "Magnitude-based pruning experiment"
    }

    # Example configuration for DeGraph pruning
    degraph_config = {
        "data_path": "./data",
        "batch_size": 16,
        "in_channels": 4,
        "out_channels": 5,
        "device": "mps",  # Use "cuda" for NVIDIA GPUs, "mps" for Apple Silicon, or "cpu"
        "sparsity_levels": [0.2, 0.4, 0.6, 0.8],
        "pruning_method": "degraph",
        "temperature": 0.1,
        "notes": "DeGraph pruning experiment"
    }

    # Choose which pruning method to run
    if len(sys.argv) > 1:
        pruning_method = sys.argv[1].lower()
        config = magnitude_config if pruning_method == 'magnitude' else degraph_config
    else:
        # Default to magnitude-based pruning
        pruning_method = 'magnitude'
        config = magnitude_config

    # Run pruning with MLflow
    prune_model_with_mlflow(
        model_path='model/dlu_net_model_best.pth',
        pruning_method=pruning_method,
        config=config,
        experiment_name="brain_segmentation_pruning",
        run_name=f"{pruning_method}_pruning"
    )
