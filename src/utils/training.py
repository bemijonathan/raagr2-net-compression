import os
import torch
from architecture.model import mean_iou, class_dice, dice_coef
from utils.mlflow import MLFlowExperiment, log_epoch_metrics, log_best_model


def resume_training(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    loss_function,
    device,
    resume_checkpoint_path,
    starting_epoch,
    num_epochs,
    checkpoint_interval=1,
    model_save_dir='model',
    experiment_name="brain_segmentation",
    tracking_uri=None,
    run_name=None,
    early_stopping_patience=10,
    min_delta=0.0001,
    val_batches=None,  # Limit validation to this many batches (None = all)
    validate_every=4,  # Validate only every N epochs
    use_amp=False      # Use automatic mixed precision
):
    """
    Resume training from a checkpoint.

    Args:
        model: The neural network model
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_function: Loss function to use
        device: Device to run training on (cuda/mps/cpu)
        resume_checkpoint_path: Path to the checkpoint to resume from
        starting_epoch: Epoch number to start from
        num_epochs: Total number of epochs to train for
        checkpoint_interval: Save checkpoints every n epochs
        model_save_dir: Directory to save model checkpoints
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking server URI
        run_name: Name for this specific MLflow run
        early_stopping_patience: Number of epochs with increasing validation loss before stopping
        min_delta: Minimum change in validation loss to be considered as improvement
        val_batches: Number of validation batches to use (None for all)
        validate_every: Run validation only every N epochs
        use_amp: Use automatic mixed precision training for faster computation

    Returns:
        model: The trained model
        best_val_loss: The best validation loss achieved
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)

    print("starting training")

    # Setup MLflow experiment
    config = {
        'optimizer': optimizer.__class__.__name__,
        'lr': optimizer.param_groups[0]['lr'],
        'scheduler': scheduler.__class__.__name__,
        'loss_function': loss_function.__class__.__name__,
        'num_epochs': num_epochs,
        'checkpoint_interval': checkpoint_interval,
        'device': str(device),
        'model_architecture': model.__class__.__name__,
        'early_stopping_patience': early_stopping_patience,
        'min_delta': min_delta,
        'val_batches': val_batches if val_batches is not None else 'all',
        'validate_every': validate_every,
        'use_amp': use_amp
    }

    # Initialize MLflow experiment
    mlflow_experiment = MLFlowExperiment(
        experiment_name=experiment_name,
        model=model,
        tracking_uri=tracking_uri,
        run_name=run_name
    )
    mlflow_experiment.log_params(config)

    # Load checkpoint
    if os.path.exists(resume_checkpoint_path):
        print(f"Loading checkpoint from {resume_checkpoint_path}")
        model.load_state_dict(torch.load(
            resume_checkpoint_path, weights_only=False))
        print(f"Resuming training from epoch {starting_epoch}")
        mlflow_experiment.log_param(
            "resumed_from_checkpoint", resume_checkpoint_path)
        mlflow_experiment.log_param("starting_epoch", starting_epoch)
    else:
        print(
            f"No checkpoint found at {resume_checkpoint_path}, starting from scratch")
        starting_epoch = 0
        mlflow_experiment.log_param("starting_from_scratch", True)

    # Load the best validation loss so far (or initialize if not available)
    best_val_loss = float('inf')

    # Early stopping variables
    epochs_without_improvement = 0
    val_loss_history = []

    # Setup automatic mixed precision if requested
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training loop
    for epoch in range(starting_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0
        class_dice_vals = {2: 0.0, 3: 0.0, 4: 0.0}

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Use mixed precision if requested
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = loss_function(outputs, masks)

                # Scale the loss and backprop
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_function(outputs, masks)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                dice = dice_coef(outputs, masks)
                iou = mean_iou(outputs, masks)

            train_loss += loss.item()
            train_dice += dice.item()
            train_iou += iou.item()

            # Compute class-wise dice scores
            for class_idx in [2, 3, 4]:
                class_dice_vals[class_idx] += class_dice(
                    outputs, masks, class_idx).item()

            # Log progress for training
            if step % 10 == 0 or step == len(train_loader):
                print(
                    f"\r{step}/{len(train_loader)} [==============================] - loss: {loss.item():.4f}", end="")
                print(
                    f"\r{step}/{len(train_loader)} [==============================] - metrics: {iou:.4f} "
                    f"c_2: {class_dice(outputs, masks, 2)} c_3: {class_dice(outputs, masks, 3)} c_4: {class_dice(outputs, masks, 4)}",
                    end=""
                )

        # Calculate average metrics for the epoch
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        for class_idx in class_dice_vals:
            class_dice_vals[class_idx] /= len(train_loader)

        print(f"\nTraining Loss: {train_loss:.4f}")

        # Run validation only every validate_every epochs or on the last epoch
        val_loss = float('inf')
        if epoch % validate_every == 0 or epoch == num_epochs - 1:
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0
            val_class_dice_vals = {2: 0.0, 3: 0.0, 4: 0.0}

            # Determine how many batches to use for validation
            batches_to_use = val_batches if val_batches is not None else len(
                val_loader)
            print(f"Running validation on {batches_to_use} batches...")

            with torch.no_grad():
                for step, (images, masks) in enumerate(val_loader, start=1):
                    # Break after processing the specified number of batches
                    if val_batches is not None and step > val_batches:
                        break

                    images = images.to(device)
                    masks = masks.to(device)

                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = loss_function(outputs, masks)
                    else:
                        outputs = model(images)
                        loss = loss_function(outputs, masks)

                    dice = dice_coef(outputs, masks)
                    iou = mean_iou(outputs, masks)

                    val_loss += loss.item()
                    val_dice += dice.item()
                    val_iou += iou.item()

                    # Compute class-wise dice scores
                    for class_idx in [2, 3, 4]:
                        val_class_dice_vals[class_idx] += class_dice(
                            outputs, masks, class_idx).item()

                    # Log progress for validation
                    if step % 10 == 0 or step == batches_to_use:
                        print(
                            f"\rValidation {step}/{batches_to_use} [==============================]"
                            f" - val_loss: {loss.item():.4f}"
                            f" - val_dice: {dice.item():.4f}",
                            end=""
                        )

                    # Log a sample of predictions to MLflow at the end of validation
                    if step == batches_to_use:
                        mlflow_experiment.log_batch_predictions(
                            images[:5].detach(),  # Take up to 5 images
                            masks[:5].detach(),
                            outputs[:5].detach(),
                            step=epoch
                        )

            # Calculate average validation metrics
            val_loss /= batches_to_use
            val_dice /= batches_to_use
            val_iou /= batches_to_use
            for class_idx in val_class_dice_vals:
                val_class_dice_vals[class_idx] /= batches_to_use

            print(
                f"\nValidation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")

            # Log metrics to MLflow
            train_metrics = {
                'train_loss': train_loss,
                'train_dice': train_dice,
                'train_iou': train_iou,
                'train_class_2_dice': class_dice_vals[2],
                'train_class_3_dice': class_dice_vals[3],
                'train_class_4_dice': class_dice_vals[4],
                'learning_rate': optimizer.param_groups[0]['lr']
            }

            val_metrics = {
                'val_loss': val_loss,
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_class_2_dice': val_class_dice_vals[2],
                'val_class_3_dice': val_class_dice_vals[3],
                'val_class_4_dice': val_class_dice_vals[4]
            }

            log_epoch_metrics(mlflow_experiment,
                              train_metrics, val_metrics, epoch)

            # Step the scheduler with validation loss
            scheduler.step(val_loss)

            # Early stopping check
            val_loss_history.append(val_loss)

            # Check if validation loss is improving
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_path = f'{model_save_dir}/dlu_net_model_best.pth'
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model with val_loss: {val_loss:.4f}")

                # Log best model to MLflow
                best_metrics = {**train_metrics, **val_metrics,
                                'best_val_loss': best_val_loss}
                log_best_model(
                    mlflow_experiment,
                    model,
                    best_model_path,
                    best_metrics,
                    registered_model_name=f"{experiment_name}_best"
                )
            else:
                epochs_without_improvement += 1
                print(
                    f"Validation loss did not improve. Epochs without improvement: {epochs_without_improvement}")

                # Check if training should be stopped
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                    # Log early stopping to MLflow
                    mlflow_experiment.log_param("stopped_early", True)
                    mlflow_experiment.log_param("stopped_epoch", epoch + 1)
                    break
        else:
            print(
                f"Skipping validation for epoch {epoch+1} (will validate every {validate_every} epochs)")
            # Without validation, we can't update the scheduler or do early stopping
            # Just log the training metrics
            train_metrics = {
                'train_loss': train_loss,
                'train_dice': train_dice,
                'train_iou': train_iou,
                'train_class_2_dice': class_dice_vals[2],
                'train_class_3_dice': class_dice_vals[3],
                'train_class_4_dice': class_dice_vals[4],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            log_epoch_metrics(mlflow_experiment, train_metrics, {}, epoch)

        # Save an intermediate checkpoint every 'checkpoint_interval' epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{model_save_dir}/dlu_net_model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Intermediate checkpoint saved at epoch {epoch+1} to '{checkpoint_path}'")

            # Log checkpoint to MLflow
            mlflow_experiment.log_state_dict(checkpoint_path)

        print('-' * 60)

    # End the MLflow run
    mlflow_experiment.end_run()

    return model, best_val_loss
