import torch
import os
import time
import gc
import numpy as np
import threading
import psutil
from typing import Dict, Any, Optional, Callable, Tuple, Union, List
from functools import wraps


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU details including name, memory capacity, and utilization.

    Returns:
        dict: Dictionary containing GPU information
    """
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": None,
        "device_name": None,
        "total_memory_gb": None,
        "allocated_memory_gb": None,
        "cached_memory_gb": None,
        "free_memory_gb": None
    }

    if gpu_info["available"]:
        gpu_info["current_device"] = torch.cuda.current_device()
        gpu_info["device_name"] = torch.cuda.get_device_name(
            gpu_info["current_device"])

        # Get memory information
        gpu_info["total_memory_gb"] = torch.cuda.get_device_properties(
            gpu_info["current_device"]).total_memory / (1024**3)
        gpu_info["allocated_memory_gb"] = torch.cuda.memory_allocated() / \
            (1024**3)
        gpu_info["cached_memory_gb"] = torch.cuda.memory_reserved() / (1024**3)
        gpu_info["free_memory_gb"] = gpu_info["total_memory_gb"] - \
            gpu_info["allocated_memory_gb"]

    return gpu_info


def get_memory_footprint() -> Dict[str, float]:
    """
    Get the current memory usage of the process.

    Returns:
        dict: Dictionary containing memory usage information
    """
    memory_info = {
        "ram_usage_gb": psutil.Process(os.getpid()).memory_info().rss / (1024**3),
        "ram_percent": psutil.Process(os.getpid()).memory_percent(),
        "system_ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "system_ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }

    return memory_info


def estimate_model_flops(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Optional[Dict[str, float]]:
    """
    Estimate FLOPs and MACs for the model using ptflops package.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (C, H, W)

    Returns:
        dict: Dictionary containing FLOPs and MACs information
    """
    try:
        from ptflops import get_model_complexity_info

        # Make a copy of the model to avoid any side effects
        import copy
        model_copy = copy.deepcopy(model)

        # Convert input shape if needed
        if len(input_shape) > 3:
            input_shape = tuple(input_shape[1:])  # Remove batch dimension

        macs, params = get_model_complexity_info(
            model_copy,
            input_shape,
            as_strings=False,
            print_per_layer_stat=False
        )

        # FLOPs is roughly 2x MACs for most operations
        flops = 2 * macs

        return {
            "flops": flops,
            "flops_billions": flops / 1e9,
            "macs": macs,
            "macs_billions": macs / 1e9,
            "params_millions": params / 1e6
        }
    except ImportError:
        print("ptflops package not installed. Install with: pip install ptflops")
        return None
    except Exception as e:
        print(f"Error estimating FLOPs: {e}")
        return None


class MemoryTracker:
    """
    Class to track memory usage during execution
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracking data"""
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.peak_gpu_memory = 0 if torch.cuda.is_available() else None
        self.initial_gpu = get_gpu_info() if torch.cuda.is_available() else None
        self.initial_memory = get_memory_footprint()
        self.final_gpu = None
        self.final_memory = None
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start memory tracking"""
        # Clear memory before starting
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Reset tracking data
        self.reset()

        # Get initial memory usage
        self.initial_gpu = get_gpu_info() if torch.cuda.is_available() else None
        self.initial_memory = get_memory_footprint()

        # Set start time
        self.start_time = time.time()

    def sample(self):
        """Take a memory sample"""
        # Get current GPU memory
        if torch.cuda.is_available():
            current_gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            self.peak_gpu_memory = max(
                self.peak_gpu_memory, current_gpu_allocated)
            self.gpu_memory_samples.append(current_gpu_allocated)

        # Get RAM usage
        current_memory = get_memory_footprint()
        self.memory_samples.append(current_memory["ram_usage_gb"])

    def stop(self):
        """Stop memory tracking and compute statistics"""
        # Final memory measurements
        self.final_gpu = get_gpu_info() if torch.cuda.is_available() else None
        self.final_memory = get_memory_footprint()
        self.end_time = time.time()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics

        Returns:
            dict: Dictionary with memory statistics
        """
        if not self.memory_samples:
            return {"error": "No memory samples collected"}

        stats = {
            "peak_ram_gb": max(self.memory_samples) if self.memory_samples else 0,
            "avg_ram_gb": sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            "initial_ram_gb": self.initial_memory["ram_usage_gb"],
            "final_ram_gb": self.final_memory["ram_usage_gb"],
            "ram_change_gb": self.final_memory["ram_usage_gb"] - self.initial_memory["ram_usage_gb"],
            "execution_time_sec": self.end_time - self.start_time
        }

        if torch.cuda.is_available():
            stats.update({
                "peak_gpu_gb": self.peak_gpu_memory,
                "avg_gpu_gb": sum(self.gpu_memory_samples) / len(self.gpu_memory_samples) if self.gpu_memory_samples else 0,
                "initial_gpu_gb": self.initial_gpu["allocated_memory_gb"],
                "final_gpu_gb": self.final_gpu["allocated_memory_gb"],
                "gpu_change_gb": self.final_gpu["allocated_memory_gb"] - self.initial_gpu["allocated_memory_gb"]
            })

        return stats


# Global tracker instance
memory_tracker = MemoryTracker()


def track_memory_usage(func: Callable) -> Callable:
    """
    Decorator to track memory usage during function execution.
    Pure data collection without any printing or side effects.

    Args:
        func: Function to track

    Returns:
        Wrapped function with memory tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start tracking
        memory_tracker.start()

        # Start memory sampling thread
        stop_sampling = threading.Event()

        def sample_memory():
            while not stop_sampling.is_set():
                memory_tracker.sample()
                # Sleep to avoid too frequent sampling
                time.sleep(0.1)

        # Start sampling thread
        sampling_thread = threading.Thread(target=sample_memory)
        sampling_thread.daemon = True
        sampling_thread.start()

        try:
            # Execute the function
            result = func(*args, **kwargs)

            # Stop memory sampling
            stop_sampling.set()
            if sampling_thread.is_alive():
                sampling_thread.join(timeout=1.0)

            # Get final measurements
            memory_tracker.stop()

            # Get memory statistics
            memory_stats = memory_tracker.get_stats()

            # Add memory stats to result without modifying original format
            if isinstance(result, tuple) and len(result) > 0 and isinstance(result[-1], dict):
                # If last element is a dict, add to it
                result_list = list(result)
                result_list[-1]["memory_stats"] = memory_stats
                return tuple(result_list)
            elif isinstance(result, dict):
                # If result is a dict, add to it
                result["memory_stats"] = memory_stats
                return result
            else:
                # Otherwise, return tuple with result and stats
                return result, memory_stats

        except Exception as e:
            stop_sampling.set()
            if sampling_thread.is_alive():
                sampling_thread.join(timeout=1.0)
            raise e

    return wrapper


def print_memory_stats(stats: Dict[str, Any]):
    """
    Print memory statistics in a formatted way.
    Separates data collection from presentation.

    Args:
        stats: Memory statistics dictionary from track_memory_usage
    """
    print("\nMemory Usage Statistics:")
    print(f"  Peak RAM: {stats['peak_ram_gb']:.2f} GB")
    print(f"  Average RAM: {stats['avg_ram_gb']:.2f} GB")
    print(f"  RAM Change: {stats['ram_change_gb']:.2f} GB")
    print(f"  Execution Time: {stats['execution_time_sec']:.2f} seconds")

    if "peak_gpu_gb" in stats:
        print(f"  Peak GPU Memory: {stats['peak_gpu_gb']:.2f} GB")
        print(f"  GPU Memory Change: {stats['gpu_change_gb']:.2f} GB")


def track_and_print_memory_usage(func: Callable) -> Callable:
    """
    Decorator that tracks memory usage and automatically prints results.
    Convenient for quick debugging.

    Args:
        func: Function to track

    Returns:
        Wrapped function with memory tracking and printing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = track_memory_usage(func)(*args, **kwargs)

        # Extract memory stats
        if isinstance(result, tuple) and len(result) > 1 and isinstance(result[-1], dict) and "memory_stats" in result[-1]:
            print_memory_stats(result[-1]["memory_stats"])
        elif isinstance(result, dict) and "memory_stats" in result:
            print_memory_stats(result["memory_stats"])

        return result

    return wrapper


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate the memory usage of a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary with model memory usage details
    """
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel()
                                for p in model.parameters() if p.requires_grad)

    # Calculate memory footprint
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    return {
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "param_memory_mb": param_bytes / (1024 * 1024),
        "buffer_memory_mb": buffer_bytes / (1024 * 1024),
        "total_memory_mb": (param_bytes + buffer_bytes) / (1024 * 1024)
    }


def calculate_model_sparsity(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate the sparsity of a PyTorch model (percentage of zero weights).

    Args:
        model: PyTorch model

    Returns:
        dict: Dictionary with sparsity information
    """
    total_params = 0
    zero_params = 0

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            param_size = param.numel()
            zero_size = (param == 0).sum().item()

            total_params += param_size
            zero_params += zero_size

    if total_params == 0:
        return {"sparsity": 0.0}

    sparsity = 100.0 * zero_params / total_params

    return {
        "total_params": total_params,
        "zero_params": zero_params,
        "nonzero_params": total_params - zero_params,
        "sparsity": sparsity
    }


def print_system_info():
    """
    Print detailed system information including CPU, RAM, and GPU details.
    """
    import platform
    import sys

    print("\n===== SYSTEM INFORMATION =====")
    print(
        f"System: {platform.system()} {platform.release()} {platform.version()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")

    # CPU information
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpu_count = psutil.cpu_count(logical=True)
    print(f"CPU: {cpu_count} physical cores, {logical_cpu_count} logical cores")

    # Memory information
    mem = psutil.virtual_memory()
    print(
        f"System RAM: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")

    # GPU information
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Compute capability: {props.major}.{props.minor}")
            print(f"  - Total memory: {props.total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA available: No")

        # Check for MPS (Apple Silicon)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) available: Yes")
            print("Using Apple Silicon GPU acceleration")
        else:
            print("Using CPU only")
