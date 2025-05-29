import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm

from architecture.model import DLUNet, ReASPP3, RRCNNBlock, AttentionBlock, UpConv
from utils.load_data import get_data_loaders

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_bn_layers(model):
    """
    Find all BatchNorm layers in the DLUNet model, with a special focus on the structure
    of ReASPP3, RRCNNBlock, and AttentionBlock components.
    
    Args:
        model: DLUNet model instance
        
    Returns:
        List of tuples (name, module) for all BatchNorm2d layers
    """
    bn_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name, module))
    
    return bn_layers

def compute_channel_importance(model, bn_layers):
    """
    Compute importance scores for each channel based on BatchNorm scaling factors
    
    Args:
        model: DLUNet model
        bn_layers: List of (name, module) for BatchNorm layers
        
    Returns:
        Dictionary mapping layer names to their channel importance scores
    """
    importance_scores = {}
    
    # Compute L1-norm of gamma parameters for each BN layer
    for name, module in bn_layers:
        importance = torch.abs(module.weight.data)
        importance_scores[name] = importance
    
    return importance_scores

def add_sparsity_regularization(optimizer, model, bn_layers, weight_decay=1e-5, bn_weight_decay=1e-4):
    """
    Add L1 regularization to BatchNorm scaling factors to encourage sparsity
    
    Args:
        optimizer: PyTorch optimizer
        model: DLUNet model
        bn_layers: List of BatchNorm layers
        weight_decay: Regular L2 weight decay value
        bn_weight_decay: L1 regularization strength for BN scaling factors
        
    Returns:
        Modified optimizer
    """
    # Set regular weight decay for all parameters except BN scaling factors
    for name, param in model.named_parameters():
        if not any(bn_name in name and 'weight' in name for bn_name, _ in bn_layers):
            param.weight_decay = weight_decay
    
    # Add hooks for L1 regularization on BN scaling factors
    def update_with_l1_regularization(grad):
        return grad.add_(torch.sign(grad.data), alpha=bn_weight_decay)
    
    # Apply hooks to BN scaling factors
    for name, module in bn_layers:
        module.weight.register_hook(update_with_l1_regularization)
    
    return optimizer

def train_with_channel_regularization(model, train_loader, val_loader, epochs=20, 
                                    learning_rate=0.001, bn_weight_decay=1e-4):
    """
    Train the model with L1 regularization on BatchNorm scaling factors
    
    Args:
        model: DLUNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate
        bn_weight_decay: L1 regularization strength for BN scaling factors
        
    Returns:
        Trained model
    """
    model.to(device)
    model.train()
    
    # Find BatchNorm layers
    bn_layers = find_bn_layers(model)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Add sparsity regularization
    optimizer = add_sparsity_regularization(optimizer, model, bn_layers, bn_weight_decay=bn_weight_decay)
    
    # Loss function - Assuming segmentation task
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # L1 regularization for BN scaling factors
                l1_reg = 0
                for name, module in bn_layers:
                    l1_reg += torch.sum(torch.abs(module.weight))
                
                # Add regularization to loss
                loss += bn_weight_decay * l1_reg
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    return best_model if best_model is not None else model

def determine_pruning_thresholds(importance_scores, target_ratio=0.3):
    """
    Determine thresholds for pruning based on importance scores
    
    Args:
        importance_scores: Dictionary of importance scores per layer
        target_ratio: Overall target pruning ratio
        
    Returns:
        Dictionary of pruning thresholds per layer
    """
    # Flatten all scores into a single tensor
    all_scores = torch.cat([scores.view(-1) for scores in importance_scores.values()])
    
    # Find the threshold that achieves the target pruning ratio
    if target_ratio > 0:
        k = int(all_scores.numel() * target_ratio)
        if k > 0:
            threshold = torch.kthvalue(all_scores, k).values.item()
        else:
            threshold = 0.0
    else:
        threshold = 0.0
    
    # Adjust thresholds based on module type
    thresholds = {}
    for name, scores in importance_scores.items():
        # We can customize thresholds by module type
        if 'enc' in name and 'final_conv' in name:
            # Encoder final layers - less aggressive
            thresholds[name] = threshold * 0.8
        elif 'enc' in name:
            # Encoder layers
            thresholds[name] = threshold * 1.0
        elif 'dec' in name:
            # Decoder layers
            thresholds[name] = threshold * 1.0
        elif 'att' in name:
            # Attention blocks - less aggressive
            thresholds[name] = threshold * 0.7
        else:
            # Default
            thresholds[name] = threshold
    
    return thresholds

def create_pruning_masks(model, importance_scores, thresholds):
    """
    Create binary masks for pruning channels
    
    Args:
        model: DLUNet model
        importance_scores: Dictionary of importance scores
        thresholds: Dictionary of pruning thresholds
        
    Returns:
        Dictionary of binary masks for each layer
    """
    masks = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if name in importance_scores and name in thresholds:
                # Create binary mask: 1 for channels to keep, 0 for channels to prune
                mask = (importance_scores[name] > thresholds[name]).float()
                masks[name] = mask
    
    return masks

def count_pruned_channels(masks):
    """
    Count number of channels to be pruned
    
    Args:
        masks: Dictionary of binary masks
        
    Returns:
        Dictionary with pruning statistics
    """
    stats = {
        'total_channels': 0,
        'pruned_channels': 0,
        'pruning_ratio': 0.0,
        'per_layer': {}
    }
    
    for name, mask in masks.items():
        total = mask.numel()
        pruned = total - torch.sum(mask).item()
        ratio = pruned / total if total > 0 else 0
        
        stats['total_channels'] += total
        stats['pruned_channels'] += pruned
        stats['per_layer'][name] = {
            'total': total,
            'pruned': pruned,
            'ratio': ratio
        }
    
    if stats['total_channels'] > 0:
        stats['pruning_ratio'] = stats['pruned_channels'] / stats['total_channels']
    
    return stats

def create_pruned_model(model, bn_layers, masks):
    """
    Create a new DLUNet model with pruned channels based on masks
    
    Args:
        model: Original DLUNet model
        bn_layers: List of BatchNorm layers
        masks: Dictionary of binary masks
        
    Returns:
        Pruned model with new structure
    """
    # Create mapping from BN layer to its channels to keep
    channels_to_keep = {}
    for name, module in bn_layers:
        if name in masks:
            mask = masks[name]
            # Indices of channels to keep (where mask is 1)
            channels_to_keep[name] = torch.nonzero(mask).squeeze().tolist()
            # Handle case where only one channel is kept
            if not isinstance(channels_to_keep[name], list):
                channels_to_keep[name] = [channels_to_keep[name]]
    
    # Create new model with modified channel counts
    in_channels = model.enc1.conv1_1[0].in_channels
    out_channels = model.final_conv.out_channels
    
    # Extract new channel configuration
    config = extract_pruned_config(model, channels_to_keep)
    
    # Create a new model with the pruned configuration
    pruned_model = create_model_with_config(in_channels, out_channels, config)
    
    # Transfer weights from original model to pruned model
    transfer_weights(model, pruned_model, channels_to_keep)
    
    return pruned_model

def extract_pruned_config(model, channels_to_keep):
    """
    Extract channel configuration for pruned model
    
    Args:
        model: Original model
        channels_to_keep: Dictionary of channel indices to keep
        
    Returns:
        Dictionary with pruned channel counts
    """
    config = {
        'enc1': len(channels_to_keep.get('enc1.final_conv.1', [])),
        'enc2': len(channels_to_keep.get('enc2.final_conv.1', [])),
        'enc3': len(channels_to_keep.get('enc3.final_conv.1', [])),
        'enc4': len(channels_to_keep.get('enc4.final_conv.1', [])),
        'enc5': len(channels_to_keep.get('enc5.final_conv.1', [])),
    }
    
    # If no channels are selected (empty list), use a minimum number
    for key in config:
        if config[key] == 0:
            # Use at least 25% of original channels or 4, whichever is larger
            original_channels = getattr(model, key).final_conv[1].num_features
            config[key] = max(4, int(original_channels * 0.25))
    
    return config

def create_model_with_config(in_channels, out_channels, config):
    """
    Create a new DLUNet model with specified channel configuration
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        config: Dictionary with channel counts for each layer
        
    Returns:
        New model with specified configuration
    """
    # Create a new model with pruned channel counts
    new_model = DLUNet(in_channels, out_channels)
    
    # Modify encoder channel counts
    new_model.enc1 = ReASPP3(in_channels, config['enc1'], 3)
    new_model.enc2 = ReASPP3(config['enc1'], config['enc2'], 3)
    new_model.enc3 = ReASPP3(config['enc2'], config['enc3'], 3)
    new_model.enc4 = ReASPP3(config['enc3'], config['enc4'], 3)
    new_model.enc5 = ReASPP3(config['enc4'], config['enc5'], 3)
    
    # Modify decoder channel counts
    new_model.up5 = UpConv(config['enc5'], config['enc4'])
    new_model.att5 = AttentionBlock(config['enc4'], config['enc4'], config['enc4'] // 2)
    new_model.dec5 = RRCNNBlock(config['enc4'] * 2, config['enc4'])
    
    new_model.up4 = UpConv(config['enc4'], config['enc3'])
    new_model.att4 = AttentionBlock(config['enc3'], config['enc3'], config['enc3'] // 2)
    new_model.dec4 = RRCNNBlock(config['enc3'] * 2, config['enc3'])
    
    new_model.up3 = UpConv(config['enc3'], config['enc2'])
    new_model.att3 = AttentionBlock(config['enc2'], config['enc2'], config['enc2'] // 2)
    new_model.dec3 = RRCNNBlock(config['enc2'] * 2, config['enc2'])
    
    new_model.up2 = UpConv(config['enc2'], config['enc1'])
    new_model.att2 = AttentionBlock(config['enc1'], config['enc1'], config['enc1'] // 2)
    new_model.dec2 = RRCNNBlock(config['enc1'] * 2, config['enc1'])
    
    # Output layer
    new_model.final_conv = nn.Conv2d(config['enc1'], out_channels, kernel_size=1)
    
    return new_model

def transfer_weights(source_model, target_model, channels_to_keep):
    """
    Transfer weights from source model to pruned target model
    
    Args:
        source_model: Original model
        target_model: Pruned model with new structure
        channels_to_keep: Dictionary of channel indices to keep
        
    Returns:
        target_model with transferred weights
    """
    # This is a complex process that would need to be implemented
    # based on specific module structures. This is a simplified version.
    # In a real implementation, this would need to be expanded to handle
    # all layer types and connections.
    
    # The implementation would depend on the exact structure
    # of the DLUNet model and would require indexing weights and biases
    # based on the channels_to_keep dictionary
    
    print("Weight transfer would need a custom implementation based on DLUNet structure")
    
    # For now, we'll leave the target model with its initialized weights
    return target_model

def fine_tune_pruned_model(model, train_loader, val_loader, epochs=10, learning_rate=0.0005):
    """
    Fine-tune the pruned model to recover accuracy
    
    Args:
        model: Pruned DLUNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        Fine-tuned model
    """
    model.to(device)
    model.train()
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_loader, desc=f"Fine-tuning {epoch+1}/{epochs}") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
    
    return best_model if best_model is not None else model

def evaluate_pruned_model(model, test_loader):
    """
    Evaluate the pruned and fine-tuned model
    
    Args:
        model: Pruned DLUNet model
        test_loader: DataLoader for test data
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.to(device)
    model.eval()
    
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Apply sigmoid to get probabilities
            outputs = torch.sigmoid(outputs)
            
            # Compute metrics
            # Assuming dice_coef and mean_iou functions are imported from architecture.model
            from architecture.model import dice_coef, mean_iou
            dice = dice_coef(outputs, targets)
            iou = mean_iou(outputs, targets)
            
            total_dice += dice.item()
            total_iou += iou.item()
            num_batches += 1
    
    results = {
        'dice_coefficient': total_dice / num_batches,
        'mean_iou': total_iou / num_batches
    }
    
    return results

def custom_dlunet_slimming(model_path, train_loader, val_loader, test_loader, 
                          pruning_ratio=0.3, bn_weight_decay=1e-4,
                          initial_epochs=15, fine_tune_epochs=10,
                          output_dir='model/slimmed'):
    """
    End-to-end pipeline for DLUNet slimming
    
    Args:
        model_path: Path to pre-trained DLUNet model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        pruning_ratio: Target ratio of channels to prune
        bn_weight_decay: L1 regularization strength for BN scaling factors
        initial_epochs: Number of epochs for initial training with regularization
        fine_tune_epochs: Number of epochs for fine-tuning after pruning
        output_dir: Directory to save pruned models
        
    Returns:
        Dictionary with results and paths to saved models
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pre-trained model
    model = torch.load(model_path, map_location=device)
    
    print("Step 1: Identifying BatchNorm layers...")
    bn_layers = find_bn_layers(model)
    print(f"Found {len(bn_layers)} BatchNorm layers")
    
    print("\nStep 2: Training with channel regularization...")
    regularized_model = train_with_channel_regularization(
        model, train_loader, val_loader, 
        epochs=initial_epochs, 
        bn_weight_decay=bn_weight_decay
    )
    
    # Save regularized model
    regularized_model_path = os.path.join(output_dir, 'regularized_model.pth')
    torch.save(regularized_model, regularized_model_path)
    print(f"Regularized model saved to {regularized_model_path}")
    
    print("\nStep 3: Computing channel importance...")
    importance_scores = compute_channel_importance(regularized_model, bn_layers)
    
    print("\nStep 4: Determining pruning thresholds...")
    thresholds = determine_pruning_thresholds(importance_scores, pruning_ratio)
    
    print("\nStep 5: Creating pruning masks...")
    masks = create_pruning_masks(regularized_model, importance_scores, thresholds)
    
    print("\nStep 6: Analyzing pruning statistics...")
    stats = count_pruned_channels(masks)
    print(f"Total channels: {stats['total_channels']}")
    print(f"Pruned channels: {stats['pruned_channels']}")
    print(f"Overall pruning ratio: {stats['pruning_ratio']:.2f}")
    
    print("\nStep 7: Creating pruned model...")
    pruned_model = create_pruned_model(regularized_model, bn_layers, masks)
    
    # Save pruned model before fine-tuning
    pruned_model_path = os.path.join(output_dir, 'pruned_model_before_tuning.pth')
    torch.save(pruned_model, pruned_model_path)
    print(f"Pruned model (before fine-tuning) saved to {pruned_model_path}")
    
    print("\nStep 8: Fine-tuning pruned model...")
    fine_tuned_model = fine_tune_pruned_model(
        pruned_model, train_loader, val_loader,
        epochs=fine_tune_epochs
    )
    
    # Save fine-tuned model
    fine_tuned_model_path = os.path.join(output_dir, 'pruned_model_fine_tuned.pth')
    torch.save(fine_tuned_model, fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")
    
    print("\nStep 9: Evaluating pruned model...")
    evaluation_results = evaluate_pruned_model(fine_tuned_model, test_loader)
    print(f"Dice Coefficient: {evaluation_results['dice_coefficient']:.4f}")
    print(f"Mean IoU: {evaluation_results['mean_iou']:.4f}")
    
    # Compare model sizes
    original_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pruned_size = sum(p.numel() for p in fine_tuned_model.parameters() if p.requires_grad)
    size_reduction_ratio = 1.0 - (pruned_size / original_size)
    
    print(f"\nOriginal model size: {original_size:,} parameters")
    print(f"Pruned model size: {pruned_size:,} parameters")
    print(f"Size reduction: {size_reduction_ratio:.2%}")
    
    # Return results
    results = {
        'original_model_size': original_size,
        'pruned_model_size': pruned_size,
        'size_reduction_ratio': size_reduction_ratio,
        'pruning_statistics': stats,
        'evaluation_results': evaluation_results,
        'model_paths': {
            'regularized': regularized_model_path,
            'pruned': pruned_model_path,
            'fine_tuned': fine_tuned_model_path
        }
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader = get_data_loaders("data", batch_size=8)
    
    results = custom_dlunet_slimming(
        model_path='model/base_model/dlu_net_model_best.pth',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pruning_ratio=0.3,  # Target 30% channel reduction
        bn_weight_decay=1e-4,
        initial_epochs=15,
        fine_tune_epochs=10,
        output_dir='model/slimmed'
    )
    
    print("\nSlimming process completed successfully!")
    print(f"Size reduction: {results['size_reduction_ratio']:.2%}")
    print(f"Final Dice Coefficient: {results['evaluation_results']['dice_coefficient']:.4f}")