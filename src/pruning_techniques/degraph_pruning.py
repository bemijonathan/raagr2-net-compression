from utils.stats import ModelPerformanceMetrics, ModelComparison
import random
from torch.utils.data import Dataset, DataLoader
from utils.load_data import BrainDataset
import torch
import torch_pruning as tp
from architecture.model import load_trained_model, class_dice
from utils.custom_loss import weighted_bce_loss
from utils.custom_metric import dice_coef, mean_iou
import os
import json

# Set device to Metal Performance Shaders (MPS) for accelerated computation on Mac
# For more on MPS: https://pytorch.org/docs/stable/notes/mps.html
device = torch.device("mps")

# ===== MODEL LOADING =====
# Load the pre-trained model from saved checkpoint
# This is the original model we'll prune
model = load_trained_model('model/dlu_net_model_epoch_11.pth')
model.to(device)  # Move model to GPU/MPS device

# Create a copy of the original model for later comparison
# This allows us to compare performance and size before and after pruning
original_model = load_trained_model('model/dlu_net_model_epoch_11.pth')
original_model.to(device)

# ===== ORIGINAL MODEL SIZE CALCULATION =====
# Calculate and display the number of trainable parameters in the original model
# numel() returns the total number of elements in a tensor
original_size = sum(p.numel()
                    for p in original_model.parameters() if p.requires_grad)
# Display in millions of parameters
print(f"Original Model size: {original_size / 1e6:.2f}M")

# ===== SAMPLE DATA PREPARATION =====
# Load a batch of validation data to use as an example input for pruning
# The pruner needs sample data to analyze the model's behavior
train_data = r"./data"
# Custom dataset for brain segmentation
val_dataset = BrainDataset(train_data, "val")
# Create data loader with batch size 16
val_loader = DataLoader(val_dataset, batch_size=16)
# Get first batch of images and masks
val_images, val_masks = next(iter(val_loader))

# Select a random sample from the batch to use as example input
random_idx = random.randint(0, val_images.shape[0]-1)
sample_image = val_images[random_idx:random_idx +
                          1].to(device)  # Single image as tensor
sample_mask = val_masks[random_idx:random_idx +
                        1].to(device)    # Corresponding mask
print(f"Sample image shape: {sample_image.shape}")

# ===== PRUNING CONFIGURATION =====
# Create a list of layers to ignore during pruning
# This is important for layers that shouldn't be pruned, like grouped convolutions
ignored_layers = []
grouped_conv_channels = {}  # Dictionary to track grouped convolution channels

# Get all the layers in your model
for name, module in model.named_modules():
    # Find your output layer and add it to ignored_layers
    if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'out_channels') and module.out_channels == 5:
        ignored_layers.append(module)

print(ignored_layers)

# Initialize the pruner with configuration parameters
# Learn more about pruning techniques: https://arxiv.org/abs/1608.08710
pruner = tp.pruner.BNScalePruner(
    model,
    example_inputs=sample_image,  # Sample input for analysis
    importance=tp.importance.MagnitudeImportance(
        p=2),  # L2 norm for determining channel importance
    # Lower importance channels will be pruned first
    pruning_ratio=0.2,           # Target to remove 30% of parameters
    max_pruning_ratio=0.3,  # Maximum pruning ratio
    # pruning_scheme=tp.pruner.function.prune_conv_in_channels,  # Commented out pruning scheme
    ignored_layers=ignored_layers,  # Layers to skip during pruning
    # Round number of channels to nearest multiple of this value
    round_to=1,
    output_transform=lambda x: x,  # Function to transform output
)

# ===== PERFORM PRUNING =====
# Execute the pruning operation
# This modifies the model in-place by removing less important channels
pruner.step()

# ===== SAVE PRUNED MODEL =====
# Remove existing pruned model file if it exists
if os.path.exists('model/pruned_dlu_net.pth'):
    os.remove('model/pruned_dlu_net.pth')
# Save the pruned model to disk
torch.save(model, 'model/pruned_dlu_net.pth')

# ===== VERIFY PRUNED MODEL =====
# Load the pruned model to ensure it was saved correctly
# weights_only=False ensures the entire model architecture is loaded
pruned_model = torch.load('model/pruned_dlu_net.pth', weights_only=False)
pruned_model.to(device)  # Move to compute device

# ===== PERFORMANCE MEASUREMENT =====
# Initialize performance metrics for both models
print("\n===== PERFORMANCE METRICS =====")
original_metrics = ModelPerformanceMetrics("Original_DLU_Net")
pruned_metrics = ModelPerformanceMetrics("Pruned_DLU_Net")

# Define metric functions for evaluation
metric_functions = {
    "mean_iou": mean_iou,
    "dice_coef": dice_coef,
    "c_2": class_dice,
    "c_3": class_dice,
    "c_4": class_dice,
}

# Create evaluation data batch for metrics computation
eval_data = (sample_image, sample_mask)

# First, do a test inference to check the model output shapes
with torch.no_grad():
    original_model.eval()
    pruned_model.eval()

    # Check output shapes
    orig_output = original_model(sample_image)
    pruned_output = pruned_model(sample_image)

    print(f"Sample mask shape: {sample_mask.shape}")
    print(f"Original model output shape: {orig_output.shape}")
    print(f"Pruned model output shape: {pruned_output.shape}")

# Extract metrics for original model
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

# Extract metrics for pruned model
print("Evaluating pruned model metrics...")
try:
    pruned_metrics.extract_metrics_from_model(
        pruned_model,
        input_data=sample_image,
        model_path='model/pruned_dlu_net.pth',
        eval_data=eval_data,
        metric_functions=metric_functions
    )
except Exception as e:
    print(f"Error evaluating pruned model: {e}")

# ===== COMPARE MODELS =====
# Initialize model comparison
print("\n===== MODEL COMPARISON =====")
model_comparison = ModelComparison(original_metrics, pruned_metrics)

# Print detailed comparison
model_comparison.print_summary()

# Record sparsity reduction
sparsity_reduction = (
    1 - pruned_metrics.metrics["num_parameters"] / original_metrics.metrics["num_parameters"]) * 100
pruned_metrics.record_sparsity_reduction(sparsity_reduction)
print(f"\nParameter reduction: {sparsity_reduction:.2f}%")

# ===== ADDITIONAL INFORMATION =====
# Set both models to evaluation mode (disables dropout, etc.)
original_model.eval()
pruned_model.eval()

# Run inference with both models to verify functionality
print("\n===== INFERENCE TEST =====")
with torch.no_grad():  # Disable gradient calculation for inference
    try:
        # Test the original model
        original_model.to(device)
        pred_mask = original_model(sample_image)
        print(f"Original model output shape: {pred_mask.shape}")

        # Test the pruned model
        pruned_model.to(device)
        pruned_pred_mask = pruned_model(sample_image)
        print(f"Pruned model output shape: {pruned_pred_mask.shape}")

        # Calculate similarity between outputs
        # This helps verify that pruning didn't dramatically change model behavior
        mse = torch.nn.MSELoss()(pred_mask, pruned_pred_mask).item()
        # Higher value means more similar outputs
        similarity = 1.0 / (1.0 + mse)
        print(f"Output similarity (normalized): {similarity:.4f}")

        # Confirmation of successful inference
        print("Both models ran successfully!")

    except Exception as e:
        # Catch and display any errors during inference
        print(f"Error during inference: {e}")

# ===== SAVE METRICS TO FILE =====
# Optionally save metrics to a file for later analysis
metrics_data = {
    "original": original_metrics.to_dict(),
    "pruned": pruned_metrics.to_dict(),
    "comparison_summary": model_comparison.compare_metrics()
}

with open('model/pruning_metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=4)
print("\nMetrics saved to model/pruning_metrics.json")


# pretrain the model
depgraph_model = load_trained_model('model/pruned_dlu_net.pth')
depgraph_model.to(device)  # Move model to GPU/MPS device
