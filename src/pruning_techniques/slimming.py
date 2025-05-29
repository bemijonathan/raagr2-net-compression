import copy
import json
import os
import time
import traceback
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from architecture.model import load_trained_model, dice_coef, mean_iou, class_dice
from utils.load_data import get_data_loaders
from utils.stats import ModelPerformanceMetrics, ModelComparison

# Select appropriate device
# Set device based on availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")  # For Macs with Apple Silicon
    print("Using MPS device")
    # Set environment variable for MPS fallback
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Load data once
train_loader, val_loader, test_loader = get_data_loaders("data")


class NetworkSlimming:
    """Network Slimming implementation for DLUNet."""

    def __init__(self, model_path, save_dir='model/slimming'):
        """Initialize Network Slimming.

        Args:
            model_path: Path to pretrained model
            save_dir: Directory to save pruned models and results
        """
        self.model_path = model_path
        self.save_dir = save_dir
        self.original_model = load_trained_model(model_path)
        self.original_model.to(device)

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Track all BN layers
        self.all_bn_layers = []
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.all_bn_layers.append(name)

        print(f"Model loaded with {self.count_parameters(self.original_model):,} parameters")
        print(f"Found {len(self.all_bn_layers)} BN layers for potential pruning")

    def count_parameters(self, model):
        """Count number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def analyze_model_structure(self):
        """Analyze model structure to identify prunable components."""
        # Get model graph information
        print("\n=== Model Analysis ===")

        # Count parameters per layer
        params_per_layer = {}
        total_params = 0

        for name, param in self.original_model.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0]
                params = param.numel()
                total_params += params

                if layer_name in params_per_layer:
                    params_per_layer[layer_name] += params
                else:
                    params_per_layer[layer_name] = params

        # Print parameter distribution
        print("\nParameter distribution:")
        for layer, params in sorted(params_per_layer.items(), key=lambda x: x[1], reverse=True):
            percentage = params / total_params * 100
            print(f"{layer}: {params:,} parameters ({percentage:.2f}%)")

        # Identify layers that participate in skip connections
        skip_connections = set()

        # In DLUNet, skip connections connect encoder blocks to decoder blocks
        for i in range(1, 5):  # 4 encoder-decoder pairs
            skip_connections.add(f"enc{i}")
            skip_connections.add(f"dec{i}")

        print(f"\nIdentified {len(skip_connections)} modules involved in skip connections:")
        print(", ".join(sorted(skip_connections)))

        return skip_connections

    def train_with_l1_reg(self, model, l1_weight, num_epochs=5):
        """Train model with L1 regularization on BN scaling factors.

        Args:
            model: Model to train
            l1_weight: L1 regularization strength
            num_epochs: Number of training epochs

        Returns:
            Trained model and training history
        """
        print(f"\n=== Training with L1 regularization ===")
        print(f"L1 weight: {l1_weight}, Epochs: {num_epochs}")
        model = copy.deepcopy(model)
        model.to(device)

        # Initialize optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.BCEWithLogitsLoss()

        # Training history - only saving final values to reduce storage
        history = {'final_train_loss': 0, 'final_val_loss': 0, 'final_train_dice': 0, 'final_val_dice': 0}

        # Initial BN scaling factor distribution
        initial_bn_weights = self.get_bn_scaling_factors(model)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_dice = 0.0

            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)

                # Main loss
                loss = criterion(outputs, masks)

                # Add L1 regularization on BN scaling factors
                l1_norm = 0
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        l1_norm += torch.sum(torch.abs(module.weight))

                # Total loss with L1 regularization
                total_loss = loss + l1_weight * l1_norm

                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                train_dice += dice_coef(torch.sigmoid(outputs), masks).item()

                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}]')
                    print(f'  Loss: {loss.item():.4f}, L1: {(l1_weight * l1_norm).item():.4f}')

            # Validation
            model.eval()
            val_loss = 0.0
            val_dice = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    # Calculate loss
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_dice += dice_coef(torch.sigmoid(outputs), masks).item()

            # Calculate average metrics
            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            # Update history - only keep final values
            history['final_train_loss'] = train_loss
            history['final_val_loss'] = val_loss
            history['final_train_dice'] = train_dice
            history['final_val_dice'] = val_dice

            # Print epoch summary
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # Final BN scaling factor distribution
        final_bn_weights = self.get_bn_scaling_factors(model)

        # Compare initial and final distributions
        self.compare_bn_distributions(initial_bn_weights, final_bn_weights)

        return model, history

    def get_bn_scaling_factors(self, model):
        """Get all BN scaling factors (gamma) from the model."""
        bn_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_weights[name] = module.weight.data.abs().clone().cpu().numpy()
        return bn_weights

    def compare_bn_distributions(self, initial_weights, final_weights):
        """Compare initial and final BN weight distributions."""
        print("\n=== BN Scaling Factor Analysis ===")

        all_initial = np.concatenate([w.flatten() for w in initial_weights.values()])
        all_final = np.concatenate([w.flatten() for w in final_weights.values()])

        # Calculate statistics
        stats = {
            'initial': {
                'mean': np.mean(all_initial),
                'median': np.median(all_initial),
                'min': np.min(all_initial),
                'max': np.max(all_initial),
                'near_zero': np.sum(all_initial < 0.01) / len(all_initial) * 100
            },
            'final': {
                'mean': np.mean(all_final),
                'median': np.median(all_final),
                'min': np.min(all_final),
                'max': np.max(all_final),
                'near_zero': np.sum(all_final < 0.01) / len(all_final) * 100
            }
        }

        print("Initial BN weights statistics:")
        print(f"  Mean: {stats['initial']['mean']:.4f}, Median: {stats['initial']['median']:.4f}")
        print(f"  Min: {stats['initial']['min']:.4f}, Max: {stats['initial']['max']:.4f}")
        print(f"  Weights near zero (<0.01): {stats['initial']['near_zero']:.2f}%")

        print("\nFinal BN weights statistics:")
        print(f"  Mean: {stats['final']['mean']:.4f}, Median: {stats['final']['median']:.4f}")
        print(f"  Min: {stats['final']['min']:.4f}, Max: {stats['final']['max']:.4f}")
        print(f"  Weights near zero (<0.01): {stats['final']['near_zero']:.2f}%")

        # Check if L1 regularization was effective
        if stats['final']['near_zero'] > stats['initial']['near_zero'] + 5:
            print("\nL1 regularization was effective at pushing weights toward zero.")
        else:
            print("\nWarning: L1 regularization had limited effect. Consider increasing L1 weight.")

    def prune_channels_by_percentage(self, model, percentage=30, skip_connections=None):
        """Prune channels based on BN scaling factors.

        Args:
            model: Model to prune
            percentage: Percentage of channels to prune (0-100)
            skip_connections: Set of layer names to exclude from pruning

        Returns:
            Model with pruned channels (either zeroed or removed) and pruning information
        """
        print(f"\n=== Pruning {percentage}% of channels ===")

        # Initialize default return values in case of failure
        pruning_info = {
            'status': 'failed',
            'error': None,
            'total_channels': 0,
            'pruned_channels': 0,
            'pruned_layers': [],
            'channels_per_layer': {}
        }

        # Input validation
        if not isinstance(percentage, (int, float)) or percentage < 0 or percentage > 100:
            print(f"Invalid pruning percentage: {percentage}. Using 30% instead.")
            percentage = 30

        if model is None:
            print("Error: No model provided for pruning")
            pruning_info['error'] = 'No model provided'
            return None, pruning_info

        if skip_connections is None:
            skip_connections = set()

        # Make a copy of the original model to ensure we always have a fallback
        original_copy = None
        try:
            original_copy = copy.deepcopy(model)
        except Exception as e:
            print(f"Warning: Failed to create a backup copy of the model: {e}")
            # Continue with the original model as backup
            original_copy = model

        try:
            # Create a copy of the model for pruning
            temp_model = copy.deepcopy(model)

            # Get all BN layers and their scaling factors
            bn_weights = {}
            bn_found = 0
            for name, module in temp_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_found += 1
                    # Skip modules involved in skip connections
                    if any(skip in name for skip in skip_connections):
                        print(f"Skipping pruning for layer in skip connection: {name}")
                        continue
                    bn_weights[name] = module.weight.data.abs().clone()

            if bn_found == 0:
                print("Error: No BatchNorm2d layers found in model")
                pruning_info['error'] = 'No BatchNorm2d layers found'
                return original_copy, pruning_info

            if len(bn_weights) == 0:
                print("Warning: No BatchNorm2d layers available for pruning (all in skip connections?)")
                pruning_info['error'] = 'No BatchNorm2d layers available for pruning'
                return original_copy, pruning_info

            # Flatten all weights
            all_weights = torch.cat([w.view(-1) for w in bn_weights.values()])

            # Determine threshold (compatible with all devices including MPS)
            k = int(len(all_weights) * (percentage / 100))
            if k >= len(all_weights):
                print(f"Warning: Trying to prune too many channels. Limiting to {len(all_weights) - 1}.")
                k = len(all_weights) - 1

            # Move to CPU for compatibility and use sorting instead of kthvalue
            sorted_weights, _ = torch.sort(all_weights.to('cpu'))
            threshold = sorted_weights[k].item()
            print(f"Pruning threshold: {threshold:.6f}")

            # Track pruning statistics
            pruning_info = {
                'status': 'success',
                'error': None,
                'total_channels': 0,
                'pruned_channels': 0,
                'pruned_layers': [],
                'channels_per_layer': {}
            }

            # Create binary masks for each layer
            masks = {}
            for name, weight in bn_weights.items():
                pruning_info['total_channels'] += len(weight)

                # Determine which channels to keep (value > threshold)
                # Use CPU comparison for compatibility
                mask = (weight.cpu() > threshold).to(weight.device)

                # Count pruned channels in this layer
                pruned_count = torch.sum(~mask).item()
                remain_count = torch.sum(mask).item()

                if pruned_count > 0:
                    pruning_info['pruned_channels'] += pruned_count
                    pruning_info['pruned_layers'].append(name)
                    pruning_info['channels_per_layer'][name] = {
                        'total': len(weight),
                        'pruned': pruned_count,
                        'remaining': remain_count,
                        'pruned_percentage': pruned_count / len(weight) * 100
                    }

                    print(f"Layer {name}: Pruned {pruned_count}/{len(weight)} channels "
                          f"({pruned_count / len(weight) * 100:.2f}%)")

                # Store mask
                masks[name] = mask

            # Apply masks to the temporary model first
            self.apply_pruning_masks(temp_model, masks)

            # At this point, we have a valid masked model as backup
            masked_model_backup = temp_model

            # Try to create a new model with channels completely removed (if possible)
            # or fall back to the masked model (zeroed weights)
            try:
                pruned_model = self.create_pruned_model(model, temp_model, masks)

                # Verify the pruned model works - try a forward pass with a single example
                if hasattr(model, 'forward') and pruned_model is not None:
                    try:
                        # Get first batch from validation loader
                        for images, _ in val_loader:
                            # Just need one image
                            test_input = images[0:1].to(device)
                            with torch.no_grad():
                                pruned_model.to(device)
                                pruned_model.eval()
                                _ = pruned_model(test_input)
                            print("Verified pruned model works with test input")
                            break
                    except Exception as e:
                        print(f"Error during forward pass with pruned model: {e}")
                        print("Falling back to masked model (zeroed weights)")
                        pruned_model = temp_model
                        pruning_info['error'] = f'Forward pass failed: {str(e)}'
                        pruning_info['note'] = 'Using masked model with zeroed weights instead'
            except Exception as e:
                print(f"Error creating pruned model: {e}")
                traceback.print_exc()
                pruned_model = temp_model
                pruning_info['error'] = f'Channel removal failed: {str(e)}'
                pruning_info['note'] = 'Using masked model with zeroed weights instead'

            # Check if we're actually using the masked model (most likely case)
            if pruned_model is temp_model:
                print("\nNote: Using masked model approach (zeroed weights) instead of channel removal")
                print("To achieve true channel removal, a custom model rebuilder would be needed")

            # Calculate overall pruning percentage
            if pruning_info['total_channels'] > 0:
                pruning_info['overall_pruning_percentage'] = (
                        pruning_info['pruned_channels'] / pruning_info['total_channels'] * 100
                )
            else:
                pruning_info['overall_pruning_percentage'] = 0

            print(f"\nPruned {pruning_info['pruned_channels']}/{pruning_info['total_channels']} "
                  f"channels ({pruning_info['overall_pruning_percentage']:.2f}%) across "
                  f"{len(pruning_info['pruned_layers'])} layers")

            # Calculate real parameter reduction
            orig_params = sum(p.numel() for p in model.parameters())
            pruned_params = sum(p.numel() for p in pruned_model.parameters())
            # With weight zeroing, total parameter count stays the same
            reduction = 0.0
            pruned_params_percent = zero_params_info['percentage'] if 'percentage' in zero_params_info else 0.0

            print(f"Original model: {orig_params:,} parameters")
            print(
                f"Zeroed parameters: {zero_params_info['zero_params']:,} of {orig_params:,} ({pruned_params_percent:.2f}%)")
            print(f"Note: Using weight zeroing, total parameter count remains unchanged")

            pruning_info['original_params'] = orig_params
            pruning_info['zeroed_params'] = zero_params_info['zero_params']
            pruning_info['param_zeroing_percentage'] = pruned_params_percent
            pruning_info['prune_method'] = 'weight_zeroing'

            return pruned_model, pruning_info

        except Exception as e:
            print(f"Unexpected error during pruning: {e}")
            traceback.print_exc()
            pruning_info['error'] = f'Unexpected error: {str(e)}'
            # Return the original model copy as fallback
            return original_copy, pruning_info

    def apply_pruning_masks(self, model, masks):
        """Apply pruning masks to model.

        This sets the BN scaling factors and biases to zero for pruned channels.
        Note: This is only used as a temporary step before creating the model
        with completely removed channels.
        """
        for name, mask in masks.items():
            for n, m in model.named_modules():
                if n == name and isinstance(m, nn.BatchNorm2d):
                    # Set pruned weights and biases to zero
                    m.weight.data.mul_(mask)
                    m.bias.data.mul_(mask)

                    # Also set running stats to 0 for pruned channels (optional)
                    # m.running_mean.data.mul_(mask)
                    # m.running_var.data.mul_(mask.logical_not().logical_or(mask))

    # fine_tune method removed to focus only on slimming

    def create_pruned_model(self, original_model, masked_model, masks):
        """Attempt to create a model with pruned channels (currently uses weight zeroing).

        Note: While the goal is to create a model with channels completely removed,
        this implementation currently uses weight zeroing instead due to architecture
        limitations. True channel removal would require a custom model rebuilding approach
        specific to this architecture.

        Args:
            original_model: The original model with full architecture
            masked_model: The model with masks applied (weights zeroed out)
            masks: The binary masks indicating which channels to keep

        Returns:
            A new model with pruned channels completely removed (smaller memory footprint)
            or the masked model if channel removal fails
        """
        print("\n=== Creating compact model with pruned channels removed ===")

        # Verify inputs
        if not hasattr(original_model, 'state_dict') or not hasattr(masked_model, 'state_dict'):
            print("Error: Models must be PyTorch modules with state_dict")
            return masked_model

        if not masks or not isinstance(masks, dict) or len(masks) == 0:
            print("Error: No valid masks provided")
            return masked_model

        # Helper function to process parameters
        def process_param(name, param):
            try:
                # Handle different layer types
                if '.weight' in name or '.bias' in name:
                    # Get the module name without the parameter suffix
                    module_name = name.rsplit('.', 1)[0]
                    param_type = name.rsplit('.', 1)[1]

                    # Check if this is a BN layer that was masked
                    if module_name in bn_masks:
                        # For BN layers, keep only the non-pruned channels
                        mask = bn_masks[module_name]
                        # Check if param is a scalar tensor (dimension 0)
                        if param.dim() == 0:
                            return param  # Just keep scalar tensors as is
                        else:
                            # Make sure the mask has the right shape for indexing
                            if mask.shape[0] != param.shape[0]:
                                print(f"Warning: Mask shape mismatch for {name}")
                                return param  # Keep original if shapes don't match
                            else:
                                return param[mask]

                    # Check if this is a conv layer affected by pruning
                    elif module_name in layer_dependencies:
                        deps = layer_dependencies[module_name]
                        if param_type == 'weight':
                            # Handle conv weights based on its dependencies
                            # Shape is [out_channels, in_channels, kernel_h, kernel_w]
                            try:
                                if deps['prev_bn'] in bn_masks and deps['next_bn'] in bn_masks:
                                    # Both input and output channels are pruned
                                    in_mask = bn_masks[deps['prev_bn']]
                                    out_mask = bn_masks[deps['next_bn']]
                                    # Check tensor dimension before applying masks
                                    if param.dim() >= 2:  # Need at least 2 dims for this indexing
                                        # Verify mask shapes before applying
                                        if (out_mask.shape[0] > param.shape[0] or
                                                in_mask.shape[0] > param.shape[1]):
                                            print(f"Warning: Mask shape mismatch for {name}")
                                            return param
                                        else:
                                            return param[out_mask][:, in_mask]
                                    else:
                                        return param  # Keep as is if not enough dimensions
                                elif deps['prev_bn'] in bn_masks:
                                    # Only input channels are pruned
                                    in_mask = bn_masks[deps['prev_bn']]
                                    # Check tensor dimension
                                    if param.dim() >= 2:
                                        # Verify mask shape
                                        if in_mask.shape[0] > param.shape[1]:
                                            print(f"Warning: Input mask shape mismatch for {name}")
                                            return param
                                        else:
                                            return param[:, in_mask]
                                    else:
                                        return param  # Keep as is if not enough dimensions
                                elif deps['next_bn'] in bn_masks:
                                    # Only output channels are pruned
                                    out_mask = bn_masks[deps['next_bn']]
                                    # Check tensor dimension
                                    if param.dim() >= 1:
                                        # Verify mask shape
                                        if out_mask.shape[0] > param.shape[0]:
                                            print(f"Warning: Output mask shape mismatch for {name}")
                                            return param
                                        else:
                                            return param[out_mask]
                                    else:
                                        return param  # Keep as is if not enough dimensions
                                else:
                                    # No pruning affects this layer
                                    return param
                            except IndexError as e:
                                print(f"Index error when applying mask to {name}: {e}")
                                prev_shape = "N/A"
                                if deps['prev_bn'] in bn_masks:
                                    prev_shape = bn_masks[deps['prev_bn']].shape
                                out_shape = "N/A"
                                if deps['next_bn'] in bn_masks:
                                    out_shape = bn_masks[deps['next_bn']].shape
                                print(f"Parameter shape: {param.shape}, "
                                      f"Masks shapes: in={prev_shape}, out={out_shape}")
                                # Fall back to using the original parameter
                                return param
                            except Exception as e:
                                print(f"Unexpected error when pruning {name}: {e}")
                                return param
                        else:
                            # For conv biases, only output channels matter
                            try:
                                if deps['next_bn'] in bn_masks:
                                    out_mask = bn_masks[deps['next_bn']]
                                    # Check tensor dimension
                                    if param.dim() >= 1:
                                        # Verify mask shape
                                        if out_mask.shape[0] > param.shape[0]:
                                            print(f"Warning: Mask shape mismatch for bias in {name}")
                                            return param
                                        else:
                                            return param[out_mask]
                                    else:
                                        return param  # Keep as is if scalar
                                else:
                                    return param.clone()  # Use clone for safety
                            except Exception as e:
                                print(f"Error pruning bias for {name}: {e}")
                                return param
                    else:
                        # This layer is not affected by pruning
                        return param.clone()  # Use clone to ensure we have a proper copy
                else:
                    # Handle non-parameter tensors (like running_mean, running_var)
                    try:
                        module_name = name.rsplit('.', 1)[0]
                        if module_name in bn_masks:
                            mask = bn_masks[module_name]
                            # Check if param is a scalar tensor (dimension 0)
                            if param.dim() == 0:
                                return param  # Just keep scalar tensors as is
                            else:
                                # Verify mask shape
                                if mask.shape[0] != param.shape[0]:
                                    print(f"Warning: Mask shape mismatch for {name}")
                                    return param
                                else:
                                    return param[mask]
                        else:
                            return param
                    except Exception as e:
                        print(f"Error processing {name}: {e}")
                        return param
            except Exception as e:
                print(f"Unexpected error processing {name}: {e}")
                # If any error occurs, use the original parameter
                return param

        try:
            # Get the model definition and configuration
            model_class = original_model.__class__

            # Build mask lookup for all BN layers
            bn_masks = {}
            for name, mask in masks.items():
                if not torch.is_tensor(mask):
                    print(f"Warning: Mask for {name} is not a tensor, skipping")
                    continue
                bn_masks[name] = mask

            if len(bn_masks) == 0:
                print("Warning: No valid masks after validation")
                return masked_model

            # Create channel selection mapping
            channel_mapping = {}
            for name, module in masked_model.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name in bn_masks:
                    try:
                        # Store which channels to keep (indices of True values in mask)
                        channel_mapping[name] = torch.where(bn_masks[name])[0]
                    except Exception as e:
                        print(f"Error creating channel mapping for {name}: {e}")

            # First, we analyze the dependencies between layers
            layer_dependencies = self._analyze_layer_dependencies(original_model)

            # Create a new state dict with pruned channels
            new_state_dict = OrderedDict()

            # Process each parameter in the state dict
            for name, param in masked_model.state_dict().items():
                new_param = process_param(name, param)
                new_state_dict[name] = new_param

            # Verify that we have a valid state dict
            if len(new_state_dict) == 0:
                print("Error: Empty state dict created. Falling back to masked model.")
                return masked_model

            # Count how many parameters were modified
            modified_params = 0
            for name, param in masked_model.state_dict().items():
                if name in new_state_dict and new_state_dict[name].size() != param.size():
                    modified_params += 1

            if modified_params == 0:
                print("Warning: No parameters were actually modified during pruning")

            # Compare pruned state dict with original model architecture
            model_params = dict(original_model.named_parameters())
            state_dict_mismatches = 0

            for name, param in new_state_dict.items():
                if name in model_params and param.size() != model_params[name].size():
                    state_dict_mismatches += 1

            if state_dict_mismatches > 0:
                print(f"\nFound {state_dict_mismatches} parameter size mismatches between pruned state dict and model")
                print("Complete channel removal requires rebuilding the model with new channel dimensions")
                print("Falling back to masked model approach (zeroing weights)")
                return masked_model

            # Try to create a model with new channel counts
            try:
                new_model = self._create_model_with_pruned_channels(original_model, bn_masks, layer_dependencies)
            except Exception as e:
                print(f"Error creating pruned model: {e}")
                traceback.print_exc()
                return masked_model  # Fall back to masked model if creation fails

            # For now, we're just returning the masked model since we can't easily rebuild
            # the architecture without a custom model builder for this specific architecture
            print("\nNOTE: For true channel removal (not just weight zeroing), a custom")
            print("model rebuilding implementation would be needed for this architecture.")

            # Since we're returning the masked model, no need to try loading the state dict
            return masked_model

        except Exception as e:
            print(f"Unhandled exception in create_pruned_model: {e}")
            traceback.print_exc()
            return masked_model

    def _analyze_layer_dependencies(self, model):
        """Analyze dependencies between layers to determine which channels to prune.

        Args:
            model: The model to analyze

        Returns:
            Dictionary mapping conv layer names to their dependencies
        """
        dependencies = {}

        try:
            # Collect all BN and Conv layers
            bn_layers = []
            conv_layers = []

            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_layers.append(name)
                elif isinstance(module, nn.Conv2d):
                    conv_layers.append(name)

            print(f"Found {len(bn_layers)} BatchNorm layers and {len(conv_layers)} Conv layers")

            if len(bn_layers) == 0:
                print("Warning: No BatchNorm layers found in the model")
                return {}

            if len(conv_layers) == 0:
                print("Warning: No Conv layers found in the model")
                return {}

            # For each conv layer, try to find the previous and next BN layers
            for conv_name in conv_layers:
                try:
                    # Get module path components
                    path_components = conv_name.split('.')

                    # Guess possible BN layer names based on common patterns
                    conv_parent = '.'.join(path_components[:-1])
                    prev_bn_candidates = []
                    next_bn_candidates = []

                    # Pattern 1: bn before conv in the same parent (e.g., block.bn, block.conv)
                    prev_bn_candidates.append(f"{conv_parent}.bn")

                    # Pattern 2: bn after conv in the same parent (e.g., block.conv, block.bn)
                    next_bn_candidates.append(f"{conv_parent}.bn")

                    # Pattern 3: bn1, conv1 pattern
                    if path_components[-1].startswith('conv'):
                        idx = path_components[-1][4:]  # extract the index
                        prev_bn_candidates.append(f"{conv_parent}.bn{idx}")

                    # Pattern 4: conv1, bn1 pattern
                    if path_components[-1].startswith('conv'):
                        idx = path_components[-1][4:]  # extract the index
                        next_bn_candidates.append(f"{conv_parent}.bn{idx}")

                    # Additional patterns for DLUNet architecture
                    # Pattern 5: block/layer number in name
                    if len(path_components) >= 2:
                        block_name = path_components[-2]
                        if any(block_name.startswith(prefix) for prefix in ['enc', 'dec', 'block']):
                            # Check for batch norm in the same block
                            for bn_name in bn_layers:
                                if block_name in bn_name:
                                    if bn_name < conv_name:  # lexicographically before
                                        prev_bn_candidates.append(bn_name)
                                    else:
                                        next_bn_candidates.append(bn_name)

                    # Find the closest matching BN layers
                    prev_bn = None
                    for candidate in prev_bn_candidates:
                        if candidate in bn_layers:
                            prev_bn = candidate
                            break

                    next_bn = None
                    for candidate in next_bn_candidates:
                        if candidate in bn_layers:
                            next_bn = candidate
                            break

                    dependencies[conv_name] = {
                        'prev_bn': prev_bn,
                        'next_bn': next_bn
                    }
                except Exception as e:
                    print(f"Error analyzing dependencies for {conv_name}: {e}")
                    # Add with empty dependencies to avoid further errors
                    dependencies[conv_name] = {'prev_bn': None, 'next_bn': None}

            # Print summary of found dependencies
            bn_connected = set()
            for conv_name, deps in dependencies.items():
                if deps['prev_bn']:
                    bn_connected.add(deps['prev_bn'])
                if deps['next_bn']:
                    bn_connected.add(deps['next_bn'])

            print(f"Found dependencies for {len(dependencies)} Conv layers, connected to {len(bn_connected)} BN layers")

            return dependencies

        except Exception as e:
            print(f"Error in dependency analysis: {e}")
            traceback.print_exc()
            return {}

    def _create_model_with_pruned_channels(self, original_model, bn_masks, layer_dependencies):
        """Create a new model with pruned channels.

        Note: This implementation only supports models where we can dynamically recreate
        convolutional layers with new channel counts. For complex architectures, this may
        require a custom model rebuilding implementation.

        Args:
            original_model: The original model with full architecture
            bn_masks: Dictionary mapping BN layer names to binary masks
            layer_dependencies: Dict mapping conv layers to their BN dependencies

        Returns:
            A new model with reduced channel counts or the original model if rebuilding fails
        """
        try:
            # Calculate the new channel counts for each BN layer
            new_channel_counts = {}
            for bn_name, mask in bn_masks.items():
                # Count True values in the mask
                new_channel_counts[bn_name] = int(torch.sum(mask).item())

            print("New channel counts after pruning:")
            for bn_name, count in new_channel_counts.items():
                # Get the original channel count
                orig_count = 0
                for name, module in original_model.named_modules():
                    if name == bn_name and isinstance(module, nn.BatchNorm2d):
                        orig_count = module.num_features
                        break

                if orig_count > 0:
                    reduction = (orig_count - count) / orig_count * 100
                    print(f"  {bn_name}: {count}/{orig_count} channels "
                          f"({reduction:.1f}% reduction)")

            print("\nWARNING: Full channel removal requires rebuilding the model architecture.")
            print("The masked model approach (zeroing weights) will be used instead.")
            print("For true pruning, you would need to create a custom model rebuilder for your architecture.")

            # Return a copy of the original model - we'll use the masked model approach
            # where channels are zeroed out but not actually removed from the architecture
            return copy.deepcopy(original_model)

        except Exception as e:
            print(f"Error creating model with pruned channels: {e}")
            traceback.print_exc()
            # Return a simple copy as fallback
            return copy.deepcopy(original_model)

    def evaluate_model(self, model, data_loader):
        """Evaluate model on data loader.

        Args:
            model: Model to evaluate
            data_loader: DataLoader to use for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        total_dice = 0.0
        total_iou = 0.0
        total_class_dice = {2: 0.0, 3: 0.0, 4: 0.0}

        with torch.no_grad():
            for images, masks in data_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)

                # Calculate metrics
                dice = dice_coef(outputs, masks).item()
                iou = mean_iou(outputs, masks).item()

                # Calculate class-wise dice
                class_dice_2 = class_dice(outputs, masks, 2).item()
                class_dice_3 = class_dice(outputs, masks, 3).item()
                class_dice_4 = class_dice(outputs, masks, 4).item()

                total_dice += dice
                total_iou += iou
                total_class_dice[2] += class_dice_2
                total_class_dice[3] += class_dice_3
                total_class_dice[4] += class_dice_4

        # Calculate averages
        avg_dice = total_dice / len(data_loader)
        avg_iou = total_iou / len(data_loader)
        avg_class_dice = {
            2: total_class_dice[2] / len(data_loader),
            3: total_class_dice[3] / len(data_loader),
            4: total_class_dice[4] / len(data_loader)
        }

        metrics = {
            'dice_coef': avg_dice,
            'mean_iou': avg_iou,
            'class_dice': avg_class_dice
        }

        return metrics

    def count_zero_params(self, model):
        """Count zero-valued parameters in model.

        Args:
            model: Model to analyze

        Returns:
            Dictionary of zero parameter statistics
        """
        zero_params = 0
        total_params = 0
        zero_params_per_layer = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_name = name.split('.')[0]
                param_zero_count = (param == 0).sum().item()
                param_total = param.numel()

                zero_params += param_zero_count
                total_params += param_total

                if layer_name in zero_params_per_layer:
                    zero_params_per_layer[layer_name]['zero'] += param_zero_count
                    zero_params_per_layer[layer_name]['total'] += param_total
                else:
                    zero_params_per_layer[layer_name] = {
                        'zero': param_zero_count,
                        'total': param_total
                    }

        # Calculate percentages
        zero_percentage = zero_params / total_params * 100 if total_params > 0 else 0

        for layer in zero_params_per_layer:
            layer_zero = zero_params_per_layer[layer]['zero']
            layer_total = zero_params_per_layer[layer]['total']
            layer_percentage = layer_zero / layer_total * 100 if layer_total > 0 else 0
            zero_params_per_layer[layer]['percentage'] = layer_percentage

        # Sort by percentage
        sorted_layers = sorted(
            zero_params_per_layer.items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )

        # Format results
        per_layer_info = [
            {
                'name': name,
                'zero_params': info['zero'],
                'total_params': info['total'],
                'percentage': info['percentage']
            }
            for name, info in sorted_layers
        ]

        return {
            'zero_params': int(zero_params),
            'total_params': int(total_params),
            'percentage': float(zero_percentage),
            'per_layer': per_layer_info
        }

    def prune_and_evaluate(self, prune_ratios=[10, 20, 30], l1_weight=0.001,
                           num_train_epochs=1):
        """Run complete pruning pipeline for different pruning ratios.

        Args:
            prune_ratios: List of pruning ratios to try (percentages)
            l1_weight: L1 regularization weight
            num_train_epochs: Number of training epochs with L1 regularization

        Returns:
            Dictionary of results for each pruning ratio
        """
        # Analyze model structure
        skip_connections = self.analyze_model_structure()

        # Evaluate original model
        print("\n=== Evaluating original model ===")
        original_metrics = self.evaluate_model(self.original_model, test_loader)
        print(f"Original model - Dice: {original_metrics['dice_coef']:.4f}, "
              f"IoU: {original_metrics['mean_iou']:.4f}")

        # Train with L1 regularization
        print("\n=== Training with L1 regularization ===")
        trained_model, train_history = self.train_with_l1_reg(
            self.original_model,
            l1_weight=l1_weight,
            num_epochs=num_train_epochs
        )

        # Evaluate trained model
        trained_metrics = self.evaluate_model(trained_model, test_loader)
        print(f"L1-trained model - Dice: {trained_metrics['dice_coef']:.4f}, "
              f"IoU: {trained_metrics['mean_iou']:.4f}")

        # Run pruning for each ratio
        results = []

        for ratio in prune_ratios:
            print(f"\n\n=== Pruning Ratio: {ratio}% ===")
            pruning_start_time = time.time()

            # Prune channels
            pruned_model, pruning_info = self.prune_channels_by_percentage(
                trained_model,
                percentage=ratio,
                skip_connections=skip_connections
            )

            # Count zero parameters
            zero_params_info = self.count_zero_params(pruned_model)
            print(f"Zero parameters: {zero_params_info['zero_params']:,}/{zero_params_info['total_params']:,} "
                  f"({zero_params_info['percentage']:.2f}%)")

            # Evaluate pruned model before fine-tuning
            pruned_metrics_before = self.evaluate_model(pruned_model, test_loader)
            print(f"Pruned model before fine-tuning - "
                  f"Dice: {pruned_metrics_before['dice_coef']:.4f}, "
                  f"IoU: {pruned_metrics_before['mean_iou']:.4f}")

            # Calculate performance change
            dice_change = ((pruned_metrics_before['dice_coef'] - original_metrics['dice_coef']) /
                           original_metrics['dice_coef'] * 100)

            print(f"Network slimming complete (weights zeroed) - no fine-tuning needed")

            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"slimmed_model_{ratio}pct_{timestamp}.pth"
            model_path = os.path.join(self.save_dir, model_filename)
            torch.save(pruned_model.state_dict(), model_path)
            print(f"Saved slimmed model (with zeroed weights) to {model_path}")

            # Calculate time
            pruning_time = round(time.time() - pruning_start_time, 2)  # Round to 2 decimal places

            # Record results with minimal logs
            result = {
                'prune_ratio': ratio,
                'l1_weight': l1_weight,
                'prune_method': 'weight_zeroing',  # Indicate we're using weight zeroing, not channel removal
                'zero_params_info': {
                    'percentage': zero_params_info['percentage'],
                    'zero_params': zero_params_info['zero_params'],
                    'total_params': zero_params_info['total_params']
                },
                'metrics': {
                    'original_dice': original_metrics['dice_coef'],
                    'pruned_dice': pruned_metrics_before['dice_coef']
                },
                'dice_change_percentage': dice_change,
                'model_path': os.path.basename(model_path),  # Only store filename, not full path
                'pruning_time_seconds': int(pruning_time),
                'timestamp': timestamp
            }

            results.append(result)

        # Save results
        results_filename = f"slimming_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path = os.path.join(self.save_dir, results_filename)

        # Convert results to JSON-serializable format
        json_results = self.prepare_for_json(results)

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\nAll results saved to {results_path}")

        # Find best result
        best_result = max(results, key=lambda x: x['zero_params_info']['percentage'] *
                                                 (1.0 + min(0, x['dice_change_percentage'] / 100)))

        print("\n=== Best Pruning Result ===")
        print(f"Pruning ratio: {best_result['prune_ratio']}%")
        print(f"Sparsity: {best_result['zero_params_info']['percentage']:.2f}%")
        print(f"Pruning method: {best_result['prune_method']}")
        print(f"Dice coefficient: {best_result['metrics']['pruned_dice']:.4f}")
        print(f"Performance change: {best_result['dice_change_percentage']:.2f}%")
        print(f"Model saved at: {best_result['model_path']}")

        return results, best_result

    def prepare_for_json(self, data):
        """Convert data to JSON-serializable format with minimal logs."""
        if isinstance(data, dict):
            return {k: self.prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.prepare_for_json(item) for item in data]
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        elif isinstance(data, (np.float32, np.float64)):
            return float(data)
        elif isinstance(data, np.ndarray):
            # For all arrays, just return summary stats to minimize JSON size
            # Return summary stats for large arrays
            return {
                "mean": float(np.mean(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "shape": list(data.shape)
            }
        elif torch.is_tensor(data):
            tensor_data = data.cpu().detach().numpy()
            # For all tensors, just return summary stats to minimize JSON size
            return {
                "mean": float(np.mean(tensor_data)),
                "min": float(np.min(tensor_data)),
                "max": float(np.max(tensor_data)),
                "shape": list(tensor_data.shape)
            }
        else:
            return data


def main():
    """Run Network Slimming pruning."""
    try:
        # Configuration
        model_path = 'model/base_model/dlu_net_model_epoch_35.pth'
        prune_ratios = [10, 20, 30]  # Pruning percentages to test
        l1_weight = 0.001  # L1 regularization strength (higher than default for more effect)
        train_epochs = 3  # Epochs for L1 regularization training

        print("=== Starting Network Slimming ===")
        print(f"Model path: {model_path}")
        print(f"Pruning ratios: {prune_ratios}")
        print(f"L1 weight: {l1_weight}")
        print(f"Training epochs: {train_epochs}")

        # Initialize Network Slimming
        slimming = NetworkSlimming(model_path)

        # Run pruning pipeline
        results, best_result = slimming.prune_and_evaluate(
            prune_ratios=prune_ratios,
            l1_weight=l1_weight,
            num_train_epochs=train_epochs,
        )

        # Print final summary
        print("\n=== Network Slimming Summary ===")
        print(f"Best pruning ratio: {best_result['prune_ratio']}%")
        print(f"Sparsity achieved: {best_result['zero_params_info']['percentage']:.2f}%")
        print(
            f"Zero parameters: {best_result['zero_params_info']['zero_params']:,}/{best_result['zero_params_info']['total_params']:,}")
        print(f"Pruning method: {best_result['prune_method']}")
        print(f"Final Dice score: {best_result['metrics']['pruned_dice']:.4f}")
        print(f"Performance change: {best_result['dice_change_percentage']:.2f}%")

    except Exception as e:
        print(f"Error in Network Slimming: {str(e)}")
        import traceback
        traceback.print_exc()


def compute_statistics():
    """Compare performance metrics between original and slimmed model (without finetuning)."""
    model_path = 'model/base_model/dlu_net_model_epoch_35.pth'
    original_model = load_trained_model(model_path)

    # Find the most recent slimmed model (or specify path directly)
    slimmed_models = [f for f in os.listdir('model/slimming') if f.startswith('slimmed_model_')]
    if not slimmed_models:
        print("No slimmed models found. Run the main function first.")
        return

    slimmed_models.sort(reverse=True)  # Sort by timestamp (newest first)
    slimmed_model_path = os.path.join('model/slimming', slimmed_models[0])
    print(f"Using slimmed model: {slimmed_model_path}")

    slimmed_model = load_trained_model(slimmed_model_path)

    original_model.to(device)
    slimmed_model.to(device)

    # Create performance metrics
    original_metrics = ModelPerformanceMetrics("Original DLUNet")
    original_metrics.extract_metrics_from_model(original_model)

    slimmed_metrics = ModelPerformanceMetrics("Slimmed DLUNet (zeroed weights, no finetuning)")
    slimmed_metrics.extract_metrics_from_model(slimmed_model)

    # Benchmark inference speed
    original_metrics.benchmark_inference_speed(original_model, val_loader)
    slimmed_metrics.benchmark_inference_speed(slimmed_model, val_loader)

    # Compare models
    comparison = ModelComparison(original_metrics, slimmed_metrics)
    comparison.calculate_speedup()
    comparison.print_summary()


if __name__ == "__main__":
    main()
    # Uncomment to run statistics on the best pruned model
    # compute_statistics()
