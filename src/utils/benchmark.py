def benchmark_model(model, val_loader, device, threshold=0.2):
    """
    Comprehensive benchmarking of a segmentation model.

    Args:
        model: PyTorch model to evaluate
        val_loader: Validation/test dataloader
        device: Computation device (cuda/cpu)
        threshold: Threshold for binary prediction (default: 0.2)

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    import time
    from collections import defaultdict

    model.eval()
    metrics = defaultdict(float)
    sample_count = 0
    total_inference_time = 0

    # Optional: Track per-sample metrics for distribution analysis
    sample_metrics = defaultdict(list)

    with torch.no_grad():
        for images, masks in val_loader:
            batch_size = images.size(0)
            images = images.to(device)
            masks = masks.to(device)

            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time

            # Apply threshold if needed
            thresholded_outputs = (outputs > threshold).float()

            # Calculate batch metrics
            batch_iou = mean_iou(thresholded_outputs, masks)
            batch_dice = dice_coef(thresholded_outputs, masks)

            # Calculate per-class metrics
            class_dices = {
                "tc_dice": class_dice(thresholded_outputs, masks, 2),  # Tumor Core
                "et_dice": class_dice(thresholded_outputs, masks, 3),  # Enhancing Tumor
                "wt_dice": class_dice(thresholded_outputs, masks, 4),  # Whole Tumor
            }

            # Additional metrics could be added here (specificity, sensitivity, etc.)

            # Accumulate metrics
            metrics["mean_iou"] += batch_iou.item() * batch_size
            metrics["dice_coef"] += batch_dice.item() * batch_size
            for key, value in class_dices.items():
                metrics[key] += value.item() * batch_size

            metrics["inference_time"] += inference_time
            sample_count += batch_size

            # Optional: Store per-sample metrics
            for i in range(batch_size):
                sample_output = outputs[i:i + 1]
                sample_mask = masks[i:i + 1]

                sample_metrics["dice"].append(dice_coef(sample_output, sample_mask).item())
                sample_metrics["tc_dice"].append(class_dice(sample_output, sample_mask, 2).item())
                sample_metrics["et_dice"].append(class_dice(sample_output, sample_mask, 3).item())
                sample_metrics["wt_dice"].append(class_dice(sample_output, sample_mask, 4).item())

    # Calculate averages
    for key in metrics:
        if key == "inference_time":
            # Average time per batch
            metrics[key] = metrics[key] / len(val_loader)
        else:
            # Average per sample
            metrics[key] = metrics[key] / sample_count

    # Add overall average across tumor regions
    metrics["avg_tumor_dice"] = (metrics["tc_dice"] + metrics["et_dice"] + metrics["wt_dice"]) / 3

    # Calculate standard deviations for analysis (optional)
    import numpy as np
    for key in sample_metrics:
        metrics[f"{key}_std"] = np.std(sample_metrics[key])
        metrics[f"{key}_min"] = np.min(sample_metrics[key])
        metrics[f"{key}_max"] = np.max(sample_metrics[key])

    return metrics


def print_benchmark_results(metrics):
    """
    Print benchmark results in a nice format.

    Args:
        metrics: Dictionary containing benchmark metrics
    """
    print("\n" + "=" * 50)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 50)

    # Print overall metrics
    print(f"\nOverall Performance:")
    print(f"  Dice Coefficient: {metrics['dice_coef']:.4f}")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")

    # Print tumor region metrics
    print(f"\nTumor Region Performance:")
    print(f"  Tumor Core (TC): {metrics['tc_dice']:.4f}")
    print(f"  Enhancing Tumor (ET): {metrics['et_dice']:.4f}")
    print(f"  Whole Tumor (WT): {metrics['wt_dice']:.4f}")
    print(f"  Average Tumor Dice: {metrics['avg_tumor_dice']:.4f}")

    # Print inference time
    print(f"\nComputation Performance:")
    print(f"  Average Inference Time per Batch: {metrics['inference_time'] * 1000:.2f} ms")

    # Print variation statistics if available
    if 'dice_std' in metrics:
        print(f"\nVariability Analysis:")
        print(
            f"  Dice Score Range: {metrics['dice_min']:.4f} - {metrics['dice_max']:.4f} (std: {metrics['dice_std']:.4f})")
        print(
            f"  TC Dice Range: {metrics['tc_dice_min']:.4f} - {metrics['tc_dice_max']:.4f} (std: {metrics['tc_dice_std']:.4f})")
        print(
            f"  ET Dice Range: {metrics['et_dice_min']:.4f} - {metrics['et_dice_max']:.4f} (std: {metrics['et_dice_std']:.4f})")
        print(
            f"  WT Dice Range: {metrics['wt_dice_min']:.4f} - {metrics['wt_dice_max']:.4f} (std: {metrics['wt_dice_std']:.4f})")

    print("=" * 50)


# Example usage:
def run_benchmark(model_path, test_loader, device):
    """Run a complete benchmark on a saved model"""
    model = load_trained_model(model_path)
    model.to(device)

    metrics = benchmark_model(model, test_loader, device)
    print_benchmark_results(metrics)

    return metrics