"""
Network Slimming Implementation for Brain Segmentation Model

This script implements the Network Slimming technique as described in:
"Learning Efficient Convolutional Networks Through Network Slimming"
by Liu et al. (ICCV 2017)
https://arxiv.org/abs/1708.06519

Network Slimming works by:
1. Adding L1 regularization on the scaling factors (gamma) of batch normalization layers
2. Identifying and pruning channels with small scaling factors
3. Fine-tuning the pruned model

Unlike magnitude-based pruning which focuses on individual weights,
network slimming targets entire channels for structured pruning,
resulting in actual speedup during inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import copy
from architecture.model import load_trained_model, DLUNet, class_dice
from utils.load_data import BrainDataset
from utils.custom_metric import dice_coef, mean_iou

# Set device to Metal Performance Shaders (MPS) for accelerated computation on Mac
device = torch.device("mps")


def get_bn_weights(model):
    """
    Get all batch normalization scaling factors (gamma) from the model.
    """
    bn_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_weights[name] = module.weight.data.abs().clone()

    return bn_weights


def l1_regularization(model, weight_decay=0.0001):
    """
    Calculate L1 regularization loss on batch normalization scaling factors.
    """
    l1_loss = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            l1_loss += torch.sum(torch.abs(module.weight))

    return weight_decay * l1_loss


def train_with_regularization(model, train_loader, val_loader, num_epochs=10,
                              learning_rate=0.001, weight_decay=0.0001,
                              save_path='model/slimmed_model.pth',
                              best_save_path='model/slimmed_model_best.pth'):
    """
    Train the model with L1 regularization on batch normalization scaling factors.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    best_val_dice = 0.0
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Main loss
            loss = criterion(outputs, masks)

            # Add L1 regularization on BN scaling factors
            l1_loss = l1_regularization(model, weight_decay)
            total_loss = loss + l1_loss

            total_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coef(outputs, masks).item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coef(outputs, masks).item()

        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_save_path)
            print(f'Saved best model with Dice: {val_dice:.4f}')

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f'Training completed. Final model saved to {save_path}')

    return history


def prune_model(model, prune_ratio=0.5, exclude_layers=None):
    """
    Prune the model based on batch normalization scaling factors.
    """
    if exclude_layers is None:
        exclude_layers = []

    # Get all BN weights
    bn_weights = get_bn_weights(model)

    # Consolidate all weights to determine global threshold
    all_weights = []
    for name, weights in bn_weights.items():
        if not any(excluded in name for excluded in exclude_layers):
            all_weights.append(weights.cpu().numpy())

    all_weights = np.concatenate([w.flatten() for w in all_weights])
    threshold = np.percentile(all_weights, prune_ratio * 100)

    print(f"Pruning threshold: {threshold:.6f}")

    # Dictionary to store pruned channels for each layer
    pruned_channels = {}

    # Identify channels to prune
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and not any(excluded in name for excluded in exclude_layers):
            weight_copy = module.weight.data.abs().clone()
            channels_to_prune = torch.where(
                weight_copy < threshold)[0].tolist()

            if len(channels_to_prune) > 0:
                pruned_channels[name] = channels_to_prune

    return model, pruned_channels


def create_pruned_model(original_model, pruned_channels):
    """
    Create a new model with pruned channels removed.
    """
    # Create a new model with the same architecture
    new_model = copy.deepcopy(original_model)

    # Get all BN layers in the model
    all_bn_modules = {}
    for name, module in new_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            all_bn_modules[name] = module

    # Prune channels
    for bn_name, channels in pruned_channels.items():
        if bn_name in all_bn_modules:
            bn_module = all_bn_modules[bn_name]
            n_channels = bn_module.weight.size(0)

            # Skip if trying to prune all channels
            if len(channels) >= n_channels:
                print(
                    f"Warning: Tried to prune all channels in {bn_name}, skipping.")
                continue

            # Create mask for channels to keep
            keep_channels = list(set(range(n_channels)) - set(channels))
            if len(keep_channels) == 0:
                print(
                    f"Warning: No channels left in {bn_name} after pruning, skipping.")
                continue

            # Update BN layer weights and biases
            bn_module.weight.data = bn_module.weight.data[keep_channels]
            bn_module.bias.data = bn_module.bias.data[keep_channels]
            bn_module.running_mean.data = bn_module.running_mean.data[keep_channels]
            bn_module.running_var.data = bn_module.running_var.data[keep_channels]
            bn_module.num_features = len(keep_channels)

            # Find corresponding Conv layers and update them
            parent_name = '.'.join(bn_name.split('.')[:-1])
            for name, module in new_model.named_modules():
                # Find Conv layer before this BN layer
                if isinstance(module, nn.Conv2d) and name == parent_name + '.0':
                    # Update output channels
                    module.out_channels = len(keep_channels)
                    module.weight.data = module.weight.data[keep_channels]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[keep_channels]

                # Find Conv/ConvTranspose layer after this BN layer
                elif (isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d)) and parent_name in name and parent_name != name:
                    # Update input channels for next layer
                    if module.groups == 1:  # Only update if not depthwise conv
                        old_weights = module.weight.data.clone()
                        module.in_channels = len(keep_channels)
                        module.weight.data = module.weight.data[:,
                                                                keep_channels]

    return new_model


def finetune_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.0005,
                   save_path='model/slimmed_model_finetuned.pth'):
    """
    Fine-tune the pruned model to recover accuracy.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coef(outputs, masks).item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_dice += dice_coef(outputs, masks).item()

        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

    # Save fine-tuned model
    torch.save(model.state_dict(), save_path)
    print(f'Fine-tuning completed. Model saved to {save_path}')

    return history


def network_slimming_pipeline(model_path, prune_ratio=0.3, batch_size=8,
                              l1_weight_decay=0.0001, exclude_layers=None):
    """
    Complete network slimming pipeline:
    1. Load model and data
    2. Train with L1 regularization on BN scaling factors
    3. Prune channels with small scaling factors
    4. Fine-tune pruned model
    5. Evaluate and compare models
    """
    # 1. Load model and data
    original_model = load_trained_model(model_path)
    original_model.to(device)

    train_data = r"./data"
    train_dataset = BrainDataset(train_data, "train")
    val_dataset = BrainDataset(train_data, "val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 2. Train with L1 regularization
    print("Step 1: Training with L1 regularization on BN scaling factors...")
    train_model = copy.deepcopy(original_model)
    train_history = train_with_regularization(
        train_model,
        train_loader,
        val_loader,
        num_epochs=5,
        weight_decay=l1_weight_decay,
        save_path='model/slimmed_model.pth',
        best_save_path='model/slimmed_model_best.pth'
    )

    # Load the best model after L1 regularization
    train_model.load_state_dict(torch.load('model/slimmed_model_best.pth'))

    # 3. Prune channels with small scaling factors
    print(f"\nStep 2: Pruning channels with prune ratio {prune_ratio}...")
    _, pruned_channels = prune_model(train_model, prune_ratio, exclude_layers)

    print(
        f"Number of batch norm layers with pruned channels: {len(pruned_channels)}")
    total_pruned_channels = sum(len(channels)
                                for channels in pruned_channels.values())
    print(f"Total pruned channels: {total_pruned_channels}")

    # 4. Create pruned model
    print("\nStep 3: Creating pruned model...")
    create_pruned_model(model, pruned_channels)

    return


if __name__ == "__main__":
    # Example usage
    model_path = 'model/dlu_net_model_best.pth'

    # Exclude output layer from pruning
    exclude_layers = ['conv_final']

    # Run the complete network slimming pipeline
    results = network_slimming_pipeline(
        model_path,
        prune_ratio=0.3,
        batch_size=8,
        l1_weight_decay=0.0001,
        exclude_layers=exclude_layers
    )
