# lets make a utility function for mlflow to record and track experiments

import os
import mlflow
import mlflow.pytorch
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import io
from PIL import Image
import warnings
from typing import Dict, Any, Optional, List, Union, Tuple


class MLFlowExperiment:
    """
    A utility class for tracking ML experiments with MLflow.

    This class provides a structured way to log model parameters, metrics,
    artifacts, and models to ensure reproducibility and easier experiment tracking.
    """

    def __init__(
        self,
        experiment_name: str,
        model: Optional[torch.nn.Module],
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the MLflow experiment.

        Args:
            experiment_name: Name of the experiment
            model: PyTorch model to track (optional)
            tracking_uri: MLflow tracking server URI (default: None, uses local filesystem)
            run_name: Name for this specific run (default: timestamp)
            nested: Whether this is a nested run within another run
            tags: Optional tags to add to the experiment
        """
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or set the experiment
        mlflow.set_experiment(experiment_name)

        # Generate run name if not provided
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set tags if not provided
        if tags is None:
            tags = {}

        # Add PyTorch version to tags
        tags["torch_version"] = torch.__version__
        if model is not None:
            tags["device"] = str(next(model.parameters()).device)
        else:
            tags["device"] = "not_specified"

        # Start the run
        self.run = mlflow.start_run(
            run_name=run_name, nested=nested, tags=tags)
        self.run_id = self.run.info.run_id
        self.model = model

        print(
            f"MLflow experiment '{experiment_name}' initialized with run ID: {self.run_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a single parameter to MLflow.

        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Step value for the metrics (default: None)
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a single metric to MLflow.

        Args:
            key: Metric name
            value: Metric value
            step: Step value for the metric (default: None)
        """
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        artifact_path: str = "model",
        sample_input: Optional[torch.Tensor] = None,
        registered_model_name: Optional[str] = None
    ) -> None:
        """
        Log the model to MLflow.

        Args:
            artifact_path: Path within the artifact location to store the model
            sample_input: Sample input tensor for model signature
            registered_model_name: Name to register the model under in the model registry
        """
        if self.model is None:
            print("No model to log. Model was initialized as None.")
            return

        if sample_input is not None:
            # Create model signature with sample input
            from mlflow.models.signature import infer_signature
            with torch.no_grad():
                sample_output = self.model(sample_input)
            signature = infer_signature(
                sample_input.cpu().numpy(), sample_output.cpu().numpy())

            mlflow.pytorch.log_model(
                self.model,
                artifact_path,
                signature=signature,
                registered_model_name=registered_model_name
            )
        else:
            mlflow.pytorch.log_model(
                self.model,
                artifact_path,
                registered_model_name=registered_model_name
            )

    def log_state_dict(self, checkpoint_path: str, artifact_path: str = "checkpoints") -> None:
        """
        Log a model state dictionary (checkpoint) to MLflow.

        Args:
            checkpoint_path: Path to the checkpoint file
            artifact_path: Path within the artifact location to store the checkpoint
        """
        mlflow.log_artifact(checkpoint_path, artifact_path)

    def log_figure(self, figure: plt.Figure, artifact_file: str) -> None:
        """
        Log a matplotlib figure to MLflow.

        Args:
            figure: Matplotlib figure to log
            artifact_file: File name for the figure
        """
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)

        # Log figure as artifact
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlflow.log_figure(figure, artifact_file)

    def log_image(self, image: Union[np.ndarray, Image.Image], artifact_file: str) -> None:
        """
        Log an image to MLflow.

        Args:
            image: Numpy array or PIL Image to log
            artifact_file: File name for the image
        """
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.ndim == 2:
                image = Image.fromarray(image, mode='L')
            elif image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image, mode='RGB')
            elif image.ndim == 3 and image.shape[2] == 4:
                image = Image.fromarray(image, mode='RGBA')

        # Log image
        mlflow.log_image(image, artifact_file)

    def log_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        class_names: List[str],
        artifact_file: str = "confusion_matrix.png"
    ) -> None:
        """
        Log a confusion matrix as a figure to MLflow.

        Args:
            conf_matrix: Confusion matrix as numpy array
            class_names: Names of the classes
            artifact_file: File name for the confusion matrix figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Set labels
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)

        # Add text annotations
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, str(conf_matrix[i, j]), ha="center", va="center")

        plt.tight_layout()
        self.log_figure(fig, artifact_file)
        plt.close(fig)

    def log_batch_predictions(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        predictions: torch.Tensor,
        max_images: int = 5,
        step: Optional[int] = None
    ) -> None:
        """
        Log a batch of image predictions (useful for segmentation visualization).

        Args:
            images: Batch of input images [B, C, H, W]
            masks: Ground truth masks [B, C, H, W]
            predictions: Model predictions [B, C, H, W]
            max_images: Maximum number of images to log
            step: Step value for the metrics (e.g., epoch number)
        """
        # Limit the number of images
        num_images = min(images.size(0), max_images)

        for i in range(num_images):
            # Create a figure with subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Plot input image (handle multi-channel inputs)
            if images.size(1) == 1:
                axs[0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
            elif images.size(1) == 3:
                img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
                axs[0].imshow(img)
            else:
                # For multi-channel medical images, use the first channel
                axs[0].imshow(images[i, 0].cpu().numpy(), cmap='gray')
            axs[0].set_title('Input Image')
            axs[0].axis('off')

            # Plot ground truth mask
            if masks.size(1) > 1:  # Multi-class segmentation
                mask_vis = torch.argmax(masks[i], dim=0).cpu().numpy()
            else:  # Binary segmentation
                mask_vis = masks[i, 0].cpu().numpy()
            axs[1].imshow(mask_vis, cmap='jet')
            axs[1].set_title('Ground Truth')
            axs[1].axis('off')

            # Plot prediction
            if predictions.size(1) > 1:  # Multi-class segmentation
                pred_vis = torch.argmax(predictions[i], dim=0).cpu().numpy()
            else:  # Binary segmentation
                pred_vis = (predictions[i, 0] > 0.5).float().cpu().numpy()
            axs[2].imshow(pred_vis, cmap='jet')
            axs[2].set_title('Prediction')
            axs[2].axis('off')

            plt.tight_layout()

            # Use step in filename if provided
            step_str = f"_step_{step}" if step is not None else ""
            self.log_figure(fig, f"prediction_sample_{i}{step_str}.png")
            plt.close(fig)

    def log_config_file(self, config_path: str) -> None:
        """
        Log a configuration file to MLflow.

        Args:
            config_path: Path to the configuration file
        """
        mlflow.log_artifact(config_path)

    def log_requirement_file(self, req_path: str = "requirements.txt") -> None:
        """
        Log the requirements file to MLflow.

        Args:
            req_path: Path to the requirements file
        """
        if os.path.exists(req_path):
            mlflow.log_artifact(req_path, "environment")
        else:
            print(f"Requirements file {req_path} not found")

    def log_pruning_details(self, pruning_details: Dict[str, Any]) -> None:
        """
        Log pruning procedure details to MLflow.

        Args:
            pruning_details: Dict containing pruning_type, original_params,
                             pruned_params, model_size_before_mb,
                             model_size_after_mb, sparsity,
                             channels_pruned, total_channels
        """
        # Log parameters and metrics for pruning
        for key, value in pruning_details.items():
            # Use parameters for descriptive values and metrics for numerical
            if isinstance(value, (str,)):
                self.log_param(key, value)
            else:
                try:
                    # numeric values logged as metrics
                    self.log_metric(key, float(value))
                except Exception:
                    self.log_param(key, str(value))

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        print("MLflow run ended")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


# Convenience functions for common use cases

def setup_experiment(
    experiment_name: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> MLFlowExperiment:
    """
    Set up an MLflow experiment with model and config parameters.

    Args:
        experiment_name: Name of the experiment
        model: PyTorch model to track
        config: Configuration dictionary with parameters to log
        run_name: Optional run name
        tracking_uri: Optional MLflow tracking server URI

    Returns:
        An initialized MLFlowExperiment
    """
    # Initialize MLflow experiment
    experiment = MLFlowExperiment(
        experiment_name=experiment_name,
        model=model,
        run_name=run_name,
        tracking_uri=tracking_uri
    )

    # Log configuration parameters
    experiment.log_params(config)

    # Log requirements file if it exists
    experiment.log_requirement_file()

    # Print experiment info
    print(
        f"MLflow experiment '{experiment_name}' set up with run ID: {experiment.run_id}")

    return experiment


def log_epoch_metrics(
    experiment: MLFlowExperiment,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    epoch: int
) -> None:
    """
    Log training and validation metrics for an epoch.

    Args:
        experiment: MLFlowExperiment instance
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        epoch: Current epoch number
    """
    # Prefix metrics with 'train_' and 'val_'
    train_metrics_prefixed = {
        f"train_{k}": v for k, v in train_metrics.items()}
    val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}

    # Combine all metrics
    all_metrics = {**train_metrics_prefixed, **val_metrics_prefixed}

    # Log all metrics
    experiment.log_metrics(all_metrics, step=epoch)


def log_best_model(
    experiment: MLFlowExperiment,
    model: torch.nn.Module,
    model_path: str,
    metrics: Dict[str, float],
    registered_model_name: Optional[str] = None
) -> None:
    """
    Log the best model and its metrics to MLflow.

    Args:
        experiment: MLFlowExperiment instance
        model: PyTorch model to log
        model_path: Path to the model checkpoint
        metrics: Dictionary of metrics for the best model
        registered_model_name: Optional name to register the model in the registry
    """
    # Log the model checkpoint
    experiment.log_state_dict(model_path)

    # Log the model architecture
    experiment.log_model(registered_model_name=registered_model_name)

    # Log metrics with 'best_' prefix
    best_metrics = {f"best_{k}": v for k, v in metrics.items()}
    experiment.log_metrics(best_metrics)

    print(f"Best model logged to MLflow with metrics: {best_metrics}")


def resume_experiment(run_id: str) -> MLFlowExperiment:
    """
    Resume an existing MLflow experiment.

    Args:
        run_id: ID of the run to resume

    Returns:
        An initialized MLFlowExperiment
    """
    # Resume the run
    run = mlflow.start_run(run_id=run_id)

    # Create a dummy experiment object
    experiment = MLFlowExperiment.__new__(MLFlowExperiment)
    experiment.run = run
    experiment.run_id = run_id
    experiment.model = None  # Will need to be set separately

    print(f"Resumed MLflow run with ID: {run_id}")

    return experiment
