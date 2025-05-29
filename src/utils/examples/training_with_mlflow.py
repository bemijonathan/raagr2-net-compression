import torch
import os
from torch.utils.data import DataLoader
from architecture.model import DLUNet
from utils.custom_loss import Weighted_BCEnDice_loss
from utils.custom_metric import dice_coef
from utils.load_data import BrainDataset
from utils.mlflow import setup_experiment, log_epoch_metrics, log_best_model
from typing import Dict, Any


def train_model_with_mlflow(
    config: Dict[str, Any],
    experiment_name: str = "brain_segmentation",
    run_name: str = None,
    tracking_uri: str = None
) -> None:
    """
    Train a brain segmentation model with MLflow tracking.

    Args:
        config: Dictionary containing training configuration
        experiment_name: Name of the MLflow experiment
        run_name: Optional name for this specific run
        tracking_uri: Optional MLflow tracking server URI
    """
    # Extract configuration values
    train_data_path = config.get("train_data_path", "./data")
    batch_size = config.get("batch_size", 16)
    num_epochs = config.get("num_epochs", 5)
    learning_rate = config.get("learning_rate", 1e-4)
    checkpoint_interval = config.get("checkpoint_interval", 1)
    in_channels = config.get("in_channels", 4)
    out_channels = config.get("out_channels", 5)
    device_name = config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu")

    # Set up device
    device = torch.device(device_name)
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs('model', exist_ok=True)

    # Initialize model
    model = DLUNet(in_channels=in_channels,
                   out_channels=out_channels).to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.1)

    # Initialize datasets and dataloaders
    train_dataset = BrainDataset(train_data_path)
    val_dataset = BrainDataset(train_data_path, "val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up MLflow experiment
    mlflow_experiment = setup_experiment(
        experiment_name=experiment_name,
        model=model,
        config=config,
        run_name=run_name,
        tracking_uri=tracking_uri
    )

    # Track best validation metrics
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_metrics = {}

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = Weighted_BCEnDice_loss(outputs, masks)
            dice = dice_coef(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            train_dice += dice.item()

            # Log progress for training
            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"\r{step}/{len(train_loader)} [==============================] - loss: {loss.item():.4f} - dice: {dice.item():.4f}", end="")

        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        print(
            f"\nTraining Loss: {train_loss:.4f}, Training Dice: {train_dice:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        # Use a batch for visualization
        vis_images = None
        vis_masks = None
        vis_outputs = None

        with torch.no_grad():
            for step, (images, masks) in enumerate(val_loader, start=1):
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)
                loss = Weighted_BCEnDice_loss(outputs, masks)
                dice = dice_coef(outputs, masks)

                # Update metrics
                val_loss += loss.item()
                val_dice += dice.item()

                # Save first batch for visualization
                if step == 1:
                    vis_images = images.detach()
                    vis_masks = masks.detach()
                    vis_outputs = outputs.detach()

                # Log progress for validation
                if step % 10 == 0 or step == len(val_loader):
                    print(
                        f"\rValidation {step}/{len(val_loader)} [==============================] - val_loss: {loss.item():.4f} - val_dice: {dice.item():.4f}", end="")

        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        print(
            f"\nValidation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Log metrics to MLflow
        epoch_metrics = {
            'train_metrics': {'loss': train_loss, 'dice': train_dice},
            'val_metrics': {'loss': val_loss, 'dice': val_dice}
        }
        log_epoch_metrics(
            experiment=mlflow_experiment,
            train_metrics={'loss': train_loss, 'dice': train_dice},
            val_metrics={'loss': val_loss, 'dice': val_dice},
            epoch=epoch
        )

        # Log sample predictions
        if vis_images is not None:
            mlflow_experiment.log_batch_predictions(
                images=vis_images,
                masks=vis_masks,
                predictions=vis_outputs,
                max_images=3,
                step=epoch
            )

        # Save checkpoint if validation improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            best_metrics = {'loss': val_loss, 'dice': val_dice}

            # Save best model
            best_model_path = 'model/dlu_net_model_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with val_loss: {val_loss:.4f}")

            # Log best model to MLflow
            log_best_model(
                experiment=mlflow_experiment,
                model=model,
                model_path=best_model_path,
                metrics=best_metrics,
                registered_model_name="brain_segmentation_dlunet"
            )

        # Save intermediate checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"model/dlu_net_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            mlflow_experiment.log_state_dict(checkpoint_path)
            print(
                f"Intermediate checkpoint saved at epoch {epoch+1} to '{checkpoint_path}'")

        print('-' * 60)

    # End MLflow run
    mlflow_experiment.end_run()
    print(
        f"Training completed. Best validation loss: {best_val_loss:.4f}, Best dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    # Example configuration
    config = {
        "train_data_path": "./data",
        "batch_size": 16,
        "num_epochs": 5,
        "learning_rate": 1e-4,
        "in_channels": 4,
        "out_channels": 5,
        "device": "mps",  # Use "cuda" for NVIDIA GPUs, "mps" for Apple Silicon, or "cpu"
        "checkpoint_interval": 1,
        # Additional parameters to track
        "model_type": "DLUNet",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "loss_function": "Weighted_BCEnDice_loss",
        "notes": "Initial training run with MLflow tracking"
    }

    # Run training with MLflow
    train_model_with_mlflow(
        config=config,
        experiment_name="brain_segmentation",
        run_name="initial_run"
    )
