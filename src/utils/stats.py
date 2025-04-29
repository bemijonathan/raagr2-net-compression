import time
import os
import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List, Tuple
from ptflops import get_model_complexity_info


class ModelPerformanceMetrics:
    """
    A class to record and track performance metrics for a model.
    """

    def __init__(self, model_name: str = "unnamed_model"):
        """
        Initialize the ModelPerformanceMetrics class.

        Args:
            model_name: Name of the model being evaluated
        """
        self.model_name = model_name
        self.metrics = {
            "inference_speed": None,  # in ms per inference
            "model_size": None,       # in MB
            "num_parameters": None,   # total number of parameters
            "parameter_size": None,   # parameter memory size in MB
            "flops": None,            # floating point operations
            "training_time": None,    # in seconds
            "sparsity_reduction": None,  # percentage
            "mean_iou": None,         # mean intersection over union
            "dice": None,             # dice coefficient
            "c_2": None,              # custom metric 2
            "c_3": None,              # custom metric 3
            "c_4": None,              # custom metric 4
        }
        self.recorded_at = None

    def benchmark_inference_speed(self, model: torch.nn.Module, input_data: torch.Tensor, iterations: int = 50) -> float:
        """
        Benchmark the inference speed of the model using real input data.

        Args:
            model: The PyTorch model to evaluate
            input_data: Input data for benchmarking
            iterations: Number of inference runs to average over

        Returns:
            Average inference time in seconds per sample
        """
        # Ensure model is in evaluation mode
        model.eval()

        try:
            # Warm up the model
            with torch.no_grad():
                _ = model(input_data)

            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_data)
            end_time = time.time()

            avg_time = (end_time - start_time) / iterations
            self.metrics["inference_speed"] = avg_time * 1000  # Convert to ms
            return avg_time
        except RuntimeError as e:
            print(f"Error during inference benchmarking: {e}")
            self.metrics["inference_speed"] = None
            return 0.0

    def measure_inference_speed(self, model: torch.nn.Module, input_shape: tuple, num_runs: int = 100) -> float:
        """
        Measure the inference speed of the model using random input.

        Args:
            model: The PyTorch model to evaluate
            input_shape: Shape of the input tensor (batch_size, channels, height, width)
            num_runs: Number of inference runs to average over

        Returns:
            Average inference time in milliseconds
        """
        # Create a dummy input
        dummy_input = torch.rand(input_shape)

        # Move to the same device as the model
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)

        # Ensure model is in evaluation mode
        model.eval()

        try:
            # Warm up the model
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)

            # Measure inference time
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = model(dummy_input)
                    end_time = time.time()
                    times.append((end_time - start_time)
                                 * 1000)  # Convert to ms

            avg_time = np.mean(times)
            self.metrics["inference_speed"] = avg_time
            return avg_time
        except RuntimeError as e:
            print(f"Error during inference speed measurement: {e}")
            self.metrics["inference_speed"] = None
            return 0.0

    def measure_model_size(self, model: torch.nn.Module) -> float:
        """
        Measure the size of the model in MB.

        Args:
            model: The PyTorch model to evaluate

        Returns:
            Model size in MB
        """
        # Save the model to a temporary file
        temp_path = "temp_model.pt"
        torch.save(model.state_dict(), temp_path)

        # Get file size in MB
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)

        # Clean up
        os.remove(temp_path)

        self.metrics["model_size"] = size_mb
        return size_mb

    def get_model_size(self, model_path: str) -> float:
        """
        Get the size of a saved model in MB.

        Args:
            model_path: Path to the saved model file

        Returns:
            Model size in MB
        """
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        self.metrics["model_size"] = size_mb
        return size_mb

    def count_parameters(self, model: torch.nn.Module) -> int:
        """
        Count the number of parameters in the model.

        Args:
            model: The PyTorch model to evaluate

        Returns:
            Total number of parameters
        """
        num_params = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        self.metrics["num_parameters"] = num_params
        # compute memory footprint of parameters in MB
        param_bytes = sum(p.numel() * p.element_size()
                          for p in model.parameters() if p.requires_grad)
        self.metrics["parameter_size"] = param_bytes / (1024 * 1024)
        return num_params

    def record_training_time(self, training_time: float) -> None:
        """
        Record the training time.

        Args:
            training_time: Training time in seconds
        """
        self.metrics["training_time"] = training_time

    def record_sparsity_reduction(self, sparsity: float) -> None:
        """
        Record the sparsity reduction percentage.

        Args:
            sparsity: Sparsity reduction percentage
        """
        self.metrics["sparsity_reduction"] = sparsity

    def record_mean_iou(self, mean_iou: float) -> None:
        """
        Record the mean IoU score.

        Args:
            mean_iou: Mean intersection over union score
        """
        self.metrics["mean_iou"] = mean_iou

    def record_dice(self, dice: float) -> None:
        """
        Record the dice coefficient.

        Args:
            dice: Dice coefficient
        """
        self.metrics["dice"] = dice

    def record_custom_metric(self, metric_name: str, value: float) -> None:
        """
        Record a custom metric.

        Args:
            metric_name: Name of the custom metric (c_2, c_3, c_4)
            value: Value of the custom metric
        """
        if metric_name in ["c_2", "c_3", "c_4"]:
            self.metrics[metric_name] = value
        else:
            raise ValueError(f"Unknown custom metric: {metric_name}")

    def record_all_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """
        Record all metrics from a dictionary.

        Args:
            metrics_dict: Dictionary containing metric names and values
        """
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key] = value

        self.recorded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all recorded metrics.

        Returns:
            Dictionary of all metrics
        """
        return self.metrics

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary including metadata.

        Returns:
            Dictionary with metrics and metadata
        """
        return {
            "model_name": self.model_name,
            "recorded_at": self.recorded_at,
            "metrics": self.metrics
        }

    def __str__(self) -> str:
        """
        String representation of the metrics.

        Returns:
            Formatted string of metrics
        """
        result = f"Model: {self.model_name}\n"
        if self.recorded_at:
            result += f"Recorded at: {self.recorded_at}\n"
        result += "Metrics:\n"

        for key, value in self.metrics.items():
            if value is not None:
                result += f"  {key}: {value}\n"

        return result

    def extract_metrics_from_model(self, model: torch.nn.Module,
                                   input_data: Optional[torch.Tensor] = None,
                                   input_shape: Optional[tuple] = None,
                                   model_path: Optional[str] = None,
                                   eval_data: Optional[tuple] = None,
                                   metric_functions: Optional[Dict[str, callable]] = None) -> Dict[str, Any]:
        """
        Extract all performance metrics from a model in a single execution.

        Args:
            model: The PyTorch model to evaluate
            input_data: Input data for benchmarking (if available)
            input_shape: Shape of the input tensor if input_data is not available
            model_path: Path to the saved model file (if available)
            eval_data: Tuple of (X_eval, y_eval) for evaluation metrics
            metric_functions: Dictionary of metric functions to evaluate

        Returns:
            Dictionary of all extracted metrics
        """
        # Count parameters
        self.count_parameters(model)

        # Measure model size
        if model_path:
            self.get_model_size(model_path)
        else:
            self.measure_model_size(model)

        # Measure inference speed
        if input_data is not None:
            self.benchmark_inference_speed(model, input_data)
        elif input_shape is not None:
            self.measure_inference_speed(model, input_shape)
        # Measure FLOPs if input_shape provided
        if input_shape is not None:
            try:
                self.measure_flops(model, input_shape)
            except ImportError:
                print("ptflops not installed, skipping flops measurement")

        # Extract evaluation metrics if eval_data is provided
        if eval_data is not None and metric_functions is not None:
            X_eval, y_eval = eval_data

            # Ensure model is in evaluation mode
            model.eval()

            # Move data to the same device as the model
            device = next(model.parameters()).device
            X_eval = X_eval.to(device)
            y_eval = y_eval.to(device)

            # Compute predictions
            with torch.no_grad():
                try:
                    y_pred = model(X_eval)

                    # Compute metrics
                    for metric_name, metric_fn in metric_functions.items():
                        try:
                            # Ensure consistent order of arguments (some metrics expect y_true first, others y_pred first)
                            if metric_name == "mean_iou":
                                result = metric_fn(y_eval, y_pred)
                            elif metric_name == "dice_coef":
                                result = metric_fn(y_eval, y_pred)
                            else:
                                result = metric_fn(
                                    y_eval, y_pred, 2 if metric_name == "c_2" else 3 if metric_name == "c_3" else 4)

                            # Record the metric
                            if isinstance(result, torch.Tensor):
                                if metric_name == "mean_iou":
                                    self.record_mean_iou(result.item())
                                elif metric_name == "dice_coef":
                                    self.record_dice(result.item())
                                elif metric_name in ["c_2", "c_3", "c_4"]:
                                    self.record_custom_metric(
                                        metric_name, result.item())
                            else:
                                if metric_name == "mean_iou":
                                    self.record_mean_iou(result)
                                elif metric_name == "dice_coef":
                                    self.record_dice(result)
                                elif metric_name in ["c_2", "c_3", "c_4"]:
                                    self.record_custom_metric(
                                        metric_name, result)

                        except Exception as e:
                            print(f"Error computing {metric_name} metric: {e}")

                    print(f"Model metrics computed with custom functions")
                except Exception as e:
                    print(f"Error during model evaluation: {e}")

        # Record timestamp
        self.recorded_at = time.strftime("%Y-%m-%d %H:%M:%S")

        return self.metrics

    def measure_flops(self, model: torch.nn.Module, input_shape: tuple) -> float:
        """
        Measure model FLOPs using ptflops.
        """
        from ptflops import get_model_complexity_info
        # ptflops expects a Python tuple of (C, H, W), not batch size or torch.Size
        # Drop batch dim if present, then cast to tuple
        if len(input_shape) > 3:
            input_res = tuple(input_shape[1:])
        else:
            input_res = tuple(input_shape)
        macs, _ = get_model_complexity_info(
            model,
            input_res,
            as_strings=False,
            print_per_layer_stat=False
        )
        flops = 2 * macs  # approximate FLOPs from MACs
        self.metrics["flops"] = flops
        return flops


class ModelComparison:
    """
    A class to compare performance metrics between two models.
    """

    def __init__(self, baseline_model: ModelPerformanceMetrics,
                 pruned_model: ModelPerformanceMetrics):
        """
        Initialize the ModelComparison class.

        Args:
            baseline_model: The baseline model metrics
            pruned_model: The pruned model metrics
        """
        self.baseline = baseline_model
        self.pruned = pruned_model

    def compare_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics between baseline and pruned models.

        Returns:
            Dictionary with comparison results
        """
        comparison = {}

        for metric_name in self.baseline.metrics:
            if (self.baseline.metrics[metric_name] is not None and
                    self.pruned.metrics[metric_name] is not None):

                baseline_value = self.baseline.metrics[metric_name]
                pruned_value = self.pruned.metrics[metric_name]

                # Calculate absolute and percentage change
                change = pruned_value - baseline_value
                change_pct = (change / baseline_value) * \
                    100 if baseline_value != 0 else 0

                comparison[metric_name] = {
                    "baseline": baseline_value,
                    "pruned": pruned_value,
                    "change": change,
                    "change_pct": change_pct
                }

        return comparison

    def print_comparison(self) -> None:
        """
        Print a formatted comparison of metrics between models.
        """
        print("\n--- Performance Comparison ---")
        print("Metric\t\tBaseline\tPruned\t\tChange")
        print("-" * 50)

        for metric_name, values in self.compare_metrics().items():
            print(
                f"{metric_name}\t\t{values['baseline']:.4f}\t{values['pruned']:.4f}\t{values['change_pct']:+.2f}%")

    def calculate_speedup(self) -> float:
        """
        Calculate the speedup factor between baseline and pruned models.

        Returns:
            Speedup factor (baseline_time / pruned_time)
        """
        if (self.baseline.metrics["inference_speed"] is not None and
                self.pruned.metrics["inference_speed"] is not None):

            # Convert to seconds
            baseline_time = self.baseline.metrics["inference_speed"] / 1000
            # Convert to seconds
            pruned_time = self.pruned.metrics["inference_speed"] / 1000

            return baseline_time / pruned_time if pruned_time > 0 else 0

        return 0

    def calculate_size_reduction(self) -> float:
        """
        Calculate the size reduction percentage between baseline and pruned models.

        Returns:
            Size reduction percentage
        """
        if (self.baseline.metrics["model_size"] is not None and
                self.pruned.metrics["model_size"] is not None):

            baseline_size = self.baseline.metrics["model_size"]
            pruned_size = self.pruned.metrics["model_size"]

            return (1 - pruned_size / baseline_size) * 100 if baseline_size > 0 else 0

        return 0

    def print_summary(self) -> None:
        """
        Print a summary of the comparison between models.
        """
        print("\n--- Model Comparison Summary ---")
        print(f"Baseline Model: {self.baseline.model_name}")
        print(f"Pruned Model: {self.pruned.model_name}")

        # Print inference speed comparison
        if (self.baseline.metrics["inference_speed"] is not None and
                self.pruned.metrics["inference_speed"] is not None):

            print(
                f"Original model inference time: {self.baseline.metrics['inference_speed']:.2f} ms/sample")
            print(
                f"Pruned model inference time: {self.pruned.metrics['inference_speed']:.2f} ms/sample")
            print(f"Speedup: {self.calculate_speedup():.2f}x")

        # Print model size comparison
        if (self.baseline.metrics["model_size"] is not None and
                self.pruned.metrics["model_size"] is not None):

            print(
                f"Original model size: {self.baseline.metrics['model_size']:.2f} MB")
            print(
                f"Pruned model size: {self.pruned.metrics['model_size']:.2f} MB")
            print(f"Size reduction: {self.calculate_size_reduction():.2f}%")
        # Print parameter size comparison
        if (self.baseline.metrics.get("parameter_size") is not None and
                self.pruned.metrics.get("parameter_size") is not None):
            print(
                f"Original parameter memory: {self.baseline.metrics['parameter_size']:.2f} MB")
            print(
                f"Pruned parameter memory: {self.pruned.metrics['parameter_size']:.2f} MB")
            print(
                f"Parameter size reduction: {self.calculate_size_reduction():.2f}%")
        # Print FLOPs comparison
        if (self.baseline.metrics.get("flops") is not None and
                self.pruned.metrics.get("flops") is not None):
            print(
                f"Original model FLOPs: {self.baseline.metrics['flops']:.2f}")
            print(
                f"Pruned model FLOPs: {self.pruned.metrics['flops']:.2f}")
            flop_reduction = (
                1 - self.pruned.metrics['flops']/self.baseline.metrics['flops'])*100
            print(f"FLOPs reduction: {flop_reduction:.2f}%")
