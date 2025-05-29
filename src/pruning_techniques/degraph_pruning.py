from utils.stats import ModelPerformanceMetrics, ModelComparison
import random
from torch.utils.data import Dataset, DataLoader
from utils.load_data import BrainDataset, get_data_loaders
import torch
import torch_pruning as tp
from architecture.model import load_trained_model, class_dice, mean_iou
from utils.custom_metric import dice_coef
import os
import json


device = torch.device("cuda")

metric_functions = {
    "mean_iou": mean_iou,
    "dice_coef": dice_coef,
    "c_2": class_dice,
    "c_3": class_dice,
    "c_4": class_dice,
}

def main():
    # Load the original model once
    original_model = load_trained_model(
        'model/base_model/dlu_net_model_epoch_35.pth')
    original_model.to(device)

    original_size = sum(p.numel()
                        for p in original_model.parameters() if p.requires_grad)
    print(f"Original Model size: {original_size / 1e6:.2f}M")

    # Prepare data
    train_loader, val_loader, test_loader = get_data_loaders("data")
    val_images, val_masks = next(iter(val_loader))

    random_idx = random.randint(0, val_images.shape[0]-1)
    sample_image = val_images[random_idx:random_idx +
                              1].to(device)  # Single image as tensor
    sample_mask = val_masks[random_idx:random_idx +
                            1].to(device)    # Corresponding mask
    print(f"Sample image shape: {sample_image.shape}")

    # Prepare metrics functions


    eval_data = (sample_image, sample_mask)

    # Extract metrics for original model once
    original_metrics = ModelPerformanceMetrics("Original_DLU_Net")
    print("Evaluating original model metrics...")
    try:
        original_metrics.extract_metrics_from_model(
            original_model,
            input_data=sample_image,
            model_path='model/dlu_net_model_best.pth',
            eval_data=eval_data,
            metric_functions=metric_functions
        )
    except Exception as e:
        print(f"Error evaluating original model: {e}")

    # Define pruning ratios to test
    pruning_ratios = [0.1, 0.2, 0.3]
    pruning_results = {}

    # Run pruning for each ratio
    for ratio in pruning_ratios:
        print(f"\n\n===== PRUNING WITH RATIO {ratio:.1f} =====")

        # Create a fresh copy of the model for this pruning ratio
        model = load_trained_model('model/base_model/dlu_net_model_epoch_35.pth')
        model.to(device)

        # Identify layers to ignore
        ignored_layers = []
        for name, module in model.named_modules():
            # Find output layer and add it to ignored_layers
            if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'out_channels') and module.out_channels == 5:
                ignored_layers.append(module)

        print(f"Ignored layers: {len(ignored_layers)}")

        # Create pruner with current ratio
        pruner = tp.pruner.BNScalePruner(
            model,
            example_inputs=sample_image,
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=ratio,
            max_pruning_ratio=ratio + 0.1,  # Allow slight flexibility
            ignored_layers=ignored_layers,
            round_to=1,
            output_transform=lambda x: x,
        )

        # Apply pruning
        pruner.step()

        # Save pruned model
        model_save_path = f'model/depgraph/pruned_dlu_net_ratio_{int(ratio*100)}.pth'
        if os.path.exists(model_save_path):
            os.remove(model_save_path)

        torch.save(model, model_save_path)
        pruned_model = torch.load(model_save_path, weights_only=False)
        pruned_model.to(device)

        # Evaluate pruned model
        print("\n===== PERFORMANCE METRICS =====")
        pruned_metrics = ModelPerformanceMetrics(
            f"Pruned_DLU_Net_Ratio_{int(ratio*100)}")

        print(f"Evaluating pruned model metrics for ratio {ratio}...")
        try:
            pruned_metrics.extract_metrics_from_model(
                pruned_model,
                input_data=sample_image,
                model_path=model_save_path,
                eval_data=eval_data,
                metric_functions=metric_functions
            )
        except Exception as e:
            print(f"Error evaluating pruned model: {e}")

        # Compare with original model
        print(f"\n===== MODEL COMPARISON FOR RATIO {ratio} =====")
        model_comparison = ModelComparison(original_metrics, pruned_metrics)
        model_comparison.print_summary()

        # Calculate parameter reduction
        sparsity_reduction = (
            1 - pruned_metrics.metrics["num_parameters"] / original_metrics.metrics["num_parameters"]) * 100
        pruned_metrics.record_sparsity_reduction(sparsity_reduction)
        print(f"\nParameter reduction: {sparsity_reduction:.2f}%")

        # Inference test
        print(f"\n===== INFERENCE TEST FOR RATIO {ratio} =====")
        with torch.no_grad():
            try:
                # Test the original model
                original_output = original_model(sample_image)

                # Test the pruned model
                pruned_output = pruned_model(sample_image)

                mse = torch.nn.MSELoss()(original_output, pruned_output).item()
                similarity = 1.0 / (1.0 + mse)
                print(f"Output similarity (normalized): {similarity:.4f}")

                # Store results
                pruning_results[f"ratio_{int(ratio*100)}"] = {
                    "metrics": pruned_metrics.to_dict(),
                    "comparison": model_comparison.compare_metrics(),
                    "similarity": similarity,
                    "parameter_reduction": sparsity_reduction
                }

            except Exception as e:
                print(f"Error during inference: {e}")

        print(f"\n===== COMPLETED PRUNING FOR RATIO {ratio} =====")

    # Save all metrics to file
    results_data = {
        "original": original_metrics.to_dict(),
        "pruning_results": pruning_results
    }

    os.makedirs('model/depgraph', exist_ok=True)
    with open('model/depgraph/multi_ratio_pruning_metrics.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    print("\nAll metrics saved to model/depgraph/multi_ratio_pruning_metrics.json")

    # Print summary comparison of all pruning ratios
    print("\n===== SUMMARY OF ALL PRUNING RATIOS =====")
    print("Ratio | Param Reduction | Similarity | Mean IoU")
    print("-----|-----------------|------------|--------")
    for ratio in pruning_ratios:
        ratio_key = f"ratio_{int(ratio*100)}"
        if ratio_key in pruning_results:
            param_red = pruning_results[ratio_key]["parameter_reduction"]
            similarity = pruning_results[ratio_key]["similarity"]
            mean_iou = pruning_results[ratio_key]["metrics"].get("mean_iou", "N/A")
            print(
                f"{ratio:.1f}   | {param_red:.2f}%          | {similarity:.4f}    | {mean_iou}")

if __name__ == "__main__":
    main()