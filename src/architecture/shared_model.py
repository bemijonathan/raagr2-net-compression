import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set GPU device if available
device = torch.device("cuda")

# Helper functions for visualization - these can remain largely the same


def devide_cls(ch2_img, ch3_img, ch4_img):
    """Divide into tumor subregions"""
    ED = ch4_img - ch2_img
    NCR = ch2_img - ch3_img
    ET = ch3_img
    return NCR, ET, ED


def color_type(NCR, ET, ED):
    RED = NCR + ET
    GREEN = ED + ET
    return RED, GREEN


def GT_color(R, G):
    """Convert to colored visualization image using PyTorch"""
    # Create zero tensor for blue channel
    if isinstance(R, torch.Tensor):
        zeros = torch.zeros_like(R)
        # Stack RGB channels
        add_img = torch.cat((R, G, zeros), dim=2)
        return add_img
    else:
        # Handle numpy arrays for backward compatibility
        add_img = np.concatenate((R, G, np.zeros((192, 192, 1))), axis=2)
        return add_img


def convert_3d(raw_img):
    """Convert single channel to RGB"""
    if isinstance(raw_img, torch.Tensor):
        # Stack the same tensor for all three channels
        Con_img = torch.cat((raw_img, raw_img, raw_img), dim=2)
        return Con_img
    else:
        # Handle numpy arrays for backward compatibility
        Con_img = np.concatenate((raw_img, raw_img, raw_img), axis=2)
        return Con_img


def result_img(X_img, Y_img):
    """Generate result visualization image using PyTorch"""
    if isinstance(X_img, torch.Tensor) and isinstance(Y_img, torch.Tensor):
        # PyTorch version
        cls1, cls2, cls3 = devide_cls(
            Y_img[..., 2:3], Y_img[..., 3:4], Y_img[..., 4:5])
        Red, Green = color_type(cls1, cls2, cls3)

        # Create inverted mask
        mask_inv = 1.0 - Y_img[..., 4:5]
        other_img = mask_inv * X_img[..., 0:1]
        GT_img = GT_color(Red, Green)

        raw = convert_3d(other_img)
        OVR_img = raw + GT_img
        return OVR_img
    else:
        # Original NumPy version for backward compatibility
        cls1, cls2, cls3 = devide_cls(
            Y_img[..., 2:3], Y_img[..., 3:4], Y_img[..., 4:5])
        Red, Green = color_type(cls1, cls2, cls3)
        mask_inv = np.reshape(cv2.bitwise_not(
            Y_img[..., 4:5]), np.shape(Y_img[..., 4:5]))
        other_img = np.reshape(cv2.bitwise_and(
            mask_inv, X_img[..., 0:1]), np.shape(mask_inv))
        GT_img = GT_color(Red, Green)

        raw = convert_3d(other_img)
        OVR_img = raw + GT_img
        return OVR_img

# PyTorch custom metrics and loss functions


def dice_coef(y_pred, y_true, smooth=1.0):
    """PyTorch version of dice coefficient"""
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def dice_loss(y_pred, y_true, smooth=1.0):
    """Dice loss function"""
    return 1 - dice_coef(y_pred, y_true, smooth)


def mean_iou(y_pred, y_true):
    """Mean IoU metric"""
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + \
        torch.sum(y_pred, dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return torch.mean(iou)


def weighted_bce_dice_loss(y_pred, y_true, weight_bce=0.5):
    """Combined weighted BCE and Dice loss"""
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_pred, y_true)
    return weight_bce * bce + (1 - weight_bce) * dice

# Custom class metrics (for specific tumor regions)


def class_dice(y_pred, y_true, class_idx, smooth=1.0):
    y_true_c = y_true[:, class_idx, :, :]
    y_pred_c = y_pred[:, class_idx, :, :]
    return dice_coef(y_pred_c, y_true_c, smooth)

# PyTorch model building blocks


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(SeparableConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class ConvOne(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvOne, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, groups=in_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class RecurrentBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv = ConvOne(in_channels, out_channels)

    def forward(self, x):
        x1 = x
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = x + x1
            x1 = self.conv(x1)
        return x1


class RRCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RRCNNBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        self.recblock1 = RecurrentBlock(out_channels, out_channels, t)
        self.recblock2 = RecurrentBlock(out_channels, out_channels, t)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.dropout(x)
        x1 = self.recblock1(x)
        x1 = self.recblock2(x1)
        return x + x1


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_int*2, kernel_size=1),
            nn.BatchNorm2d(F_int*2),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class SharedDepthwiseBlock(nn.Module):
    """
    A single “shared‐depthwise → pointwise → residual” block.
    Depthwise uses outer model’s shared weights; pointwise and
    residual projections are per‐block.
    """

    def __init__(self, in_ch, out_ch, dilation, shared_dw_weight, shared_dw_bias):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.padding = dilation

        # Reference to the shared depthwise parameters:
        self.shared_dw_weight = shared_dw_weight
        self.shared_dw_bias = shared_dw_bias

        # 1×1 pointwise to project depthwise→out_ch
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # 1×1 residual projection: in_ch→out_ch
        self.res_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        # 1) Depthwise (shared weights)
        d = F.conv2d(
            x,
            self.shared_dw_weight,
            bias=self.shared_dw_bias,
            stride=1,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.in_ch
        )  # → [B, in_ch, H, W]

        # 2) Pointwise + BN + ReLU
        y = self.pw(d)     # → [B, out_ch, H, W]
        y = self.bn(y)
        y = self.act(y)

        # 3) Residual projection and addition
        res = self.res_proj(d)  # → [B, out_ch, H, W]
        return y + res          # channel dims match

# ------------------------------------------------------------------------------


class ReASPP3(nn.Module):
    def __init__(self, in_channels, out_channels, r=3):
        super().__init__()

        # Shared depthwise (kernel=3×3)
        #   shape: (in_channels, 1, 3, 3)
        self.shared_dw_weight = nn.Parameter(
            torch.randn(in_channels, 1, 3, 3))
        self.shared_dw_bias = nn.Parameter(
            torch.randn(in_channels))

        # Four parallel blocks with increasing dilation
        self.block1 = SharedDepthwiseBlock(
            in_channels, out_channels, dilation=1,
            shared_dw_weight=self.shared_dw_weight,
            shared_dw_bias=self.shared_dw_bias
        )
        self.block2 = SharedDepthwiseBlock(
            in_channels, out_channels, dilation=r,
            shared_dw_weight=self.shared_dw_weight,
            shared_dw_bias=self.shared_dw_bias
        )
        self.block3 = SharedDepthwiseBlock(
            in_channels, out_channels, dilation=r*2,
            shared_dw_weight=self.shared_dw_weight,
            shared_dw_bias=self.shared_dw_bias
        )
        self.block4 = SharedDepthwiseBlock(
            in_channels, out_channels, dilation=r*3,
            shared_dw_weight=self.shared_dw_weight,
            shared_dw_bias=self.shared_dw_bias
        )

        # Final 1×1 conv to fuse all branches + identity
        total_channels = out_channels * 4 + in_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Apply each shared-depthwise block
        y1 = self.block1(x)
        y2 = self.block2(x)
        y3 = self.block3(x)
        y4 = self.block4(x)

        # Concatenate branch outputs + identity
        cat = torch.cat([y1, y2, y3, y4, x], dim=1)
        return self.final_conv(cat)

# Main DLU-Net model


class DLUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):
        super(DLUNet, self).__init__()

        # Encoder path
        self.enc1 = ReASPP3(in_channels, 32, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ReASPP3(32, 64, 3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ReASPP3(64, 128, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ReASPP3(128, 256, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5 = ReASPP3(256, 512, 3)

        # Decoder path
        self.up5 = UpConv(512, 256)
        self.att5 = AttentionBlock(256, 256, 128)
        self.dec5 = RRCNNBlock(512, 256)

        self.up4 = UpConv(256, 128)
        self.att4 = AttentionBlock(128, 128, 64)
        self.dec4 = RRCNNBlock(256, 128)

        self.up3 = UpConv(128, 64)
        self.att3 = AttentionBlock(64, 64, 32)
        self.dec3 = RRCNNBlock(128, 64)

        self.up2 = UpConv(64, 32)
        self.att2 = AttentionBlock(32, 32, 16)
        self.dec2 = RRCNNBlock(64, 32)

        # Final output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        e5 = self.enc5(p4)

        # Decoder path with attention and skip connections
        d5 = self.up5(e5)
        a4 = self.att5(d5, e4)
        d5 = torch.cat([a4, d5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        a3 = self.att4(d4, e3)
        d4 = torch.cat([a3, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        a2 = self.att3(d3, e2)
        d3 = torch.cat([a2, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        a1 = self.att2(d2, e1)
        d2 = torch.cat([a1, d2], dim=1)
        d2 = self.dec2(d2)

        out = self.final_conv(d2)
        return self.sigmoid(out)

# Data handling


class BrainTumorDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float().permute(
            2, 0, 1)  # [H,W,C] -> [C,H,W]
        mask = torch.from_numpy(mask).float().permute(2, 0, 1)

        if self.transform:
            # Apply transformations if needed
            pass

        return image, mask


def get_initial_model():
    """Create and return an initialized DLU-Net model"""
    try:
        model = DLUNet(in_channels=4).to(device)
        print("Model created successfully:", model)
        return model
    except Exception as e:
        print("Error creating model:", e)
        raise


def load_trained_model(model_path):
    """Load a trained model from a checkpoint file"""
    model = DLUNet(in_channels=4)
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Example training function


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-4):
    """Train the PyTorch model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, min_lr=0.00001, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = weighted_bce_dice_loss(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = weighted_bce_dice_loss(outputs, masks)
                dice = dice_coef(outputs, masks)

                val_loss += loss.item()
                val_dice += dice.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model/dlu_net_model.pth')
            print(f"Saved new best model with val_loss: {val_loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Dice: {val_dice:.4f}")

    return model

# Function to create data loaders


def create_data_loaders(X_train, Y_train, X_val, Y_val, batch_size=8):
    """Create PyTorch DataLoaders for training and validation data"""
    train_dataset = BrainTumorDataset(X_train, Y_train)
    val_dataset = BrainTumorDataset(X_val, Y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def devide_cls(ch2_img, ch3_img, ch4_img):
    """Divide into tumor subregions"""
    ED = ch4_img - ch2_img
    NCR = ch2_img - ch3_img
    ET = ch3_img
    return NCR, ET, ED


def color_type(NCR, ET, ED):
    RED = NCR + ET
    GREEN = ED + ET
    return RED, GREEN


def GT_color(R, G):
    """Convert to colored visualization image using PyTorch"""
    # Create zero tensor for blue channel
    if isinstance(R, torch.Tensor):
        zeros = torch.zeros_like(R)
        # Stack RGB channels
        add_img = torch.cat((R, G, zeros), dim=2)
        return add_img
    else:
        # Handle numpy arrays for backward compatibility
        add_img = np.concatenate((R, G, np.zeros((192, 192, 1))), axis=2)
        return add_img


def convert_3d(raw_img):
    """Convert single channel to RGB"""
    if isinstance(raw_img, torch.Tensor):
        # Stack the same tensor for all three channels
        Con_img = torch.cat((raw_img, raw_img, raw_img), dim=2)
        return Con_img
    else:
        # Handle numpy arrays for backward compatibility
        Con_img = np.concatenate((raw_img, raw_img, raw_img), axis=2)
        return Con_img


def result_img(X_img, Y_img):
    """Generate result visualization image using PyTorch"""
    if isinstance(X_img, torch.Tensor) and isinstance(Y_img, torch.Tensor):
        # PyTorch version
        cls1, cls2, cls3 = devide_cls(
            Y_img[..., 2:3], Y_img[..., 3:4], Y_img[..., 4:5])
        Red, Green = color_type(cls1, cls2, cls3)

        # Create inverted mask
        mask_inv = 1.0 - Y_img[..., 4:5]
        other_img = mask_inv * X_img[..., 0:1]
        GT_img = GT_color(Red, Green)

        raw = convert_3d(other_img)
        OVR_img = raw + GT_img
        return OVR_img
    else:
        # Original NumPy version for backward compatibility
        cls1, cls2, cls3 = devide_cls(
            Y_img[..., 2:3], Y_img[..., 3:4], Y_img[..., 4:5])
        Red, Green = color_type(cls1, cls2, cls3)
        mask_inv = np.reshape(cv2.bitwise_not(
            Y_img[..., 4:5]), np.shape(Y_img[..., 4:5]))
        other_img = np.reshape(cv2.bitwise_and(
            mask_inv, X_img[..., 0:1]), np.shape(mask_inv))
        GT_img = GT_color(Red, Green)

        raw = convert_3d(other_img)
        OVR_img = raw + GT_img
        return OVR_img


def arrange_img(X_img, Y_img, prediction):
    """Arrange images for visualization using PyTorch"""
    if isinstance(X_img, torch.Tensor) and isinstance(Y_img, torch.Tensor) and isinstance(prediction, torch.Tensor):
        # PyTorch version
        GT_img = result_img(X_img[0].cpu(), Y_img[0].cpu())
        Pre_img = result_img(X_img[0].cpu(), prediction[0].cpu(
        ) if prediction.dim() == 4 else prediction.cpu())

        # Create comparison visualizations for each tumor region
        zeros = torch.zeros((192, 192, 1), device='cpu')

        # Handle dimension mismatch by ensuring all tensors are 3D
        pred_tc = prediction[0, :, :, 2:3].cpu(
        ) if prediction.dim() == 4 else prediction[:, :, 2:3].cpu()
        pred_ec = prediction[0, :, :, 3:4].cpu(
        ) if prediction.dim() == 4 else prediction[:, :, 3:4].cpu()
        pred_wt = prediction[0, :, :, 4:5].cpu(
        ) if prediction.dim() == 4 else prediction[:, :, 4:5].cpu()

        TC = torch.cat((pred_tc, Y_img[0, :, :, 2:3].cpu(), zeros), dim=2)
        EC = torch.cat((pred_ec, Y_img[0, :, :, 3:4].cpu(), zeros), dim=2)
        WT = torch.cat((pred_wt, Y_img[0, :, :, 4:5].cpu(), zeros), dim=2)

        return GT_img, Pre_img, TC, EC, WT
    else:
        # Original NumPy version for backward compatibility
        GT_img = result_img(X_img[0], Y_img[0])
        Pre_img = result_img(X_img[0], prediction)

        TC = np.concatenate(
            (prediction[..., 2:3], Y_img[0, :, :, 2:3], np.zeros((192, 192, 1))), axis=2)
        EC = np.concatenate(
            (prediction[..., 3:4], Y_img[0, :, :, 3:4], np.zeros((192, 192, 1))), axis=2)
        WT = np.concatenate(
            (prediction[..., 4:5], Y_img[0, :, :, 4:5], np.zeros((192, 192, 1))), axis=2)

        return GT_img, Pre_img, TC, EC, WT


def evaluate_dice_scores(model, X, Y):
    """Calculate dice scores for tumor regions"""
    model.eval()
    with torch.no_grad():
        # Convert numpy arrays to PyTorch tensors with correct dimension order
        # Change from [batch, height, width, channels] to [batch, channels, height, width]
        X_tensor = torch.from_numpy(X).float().permute(0, 3, 1, 2).to(device)
        Y_tensor = torch.from_numpy(Y).float().permute(0, 3, 1, 2).to(device)

        # Get predictions
        outputs = model(X_tensor)

        # Calculate dice scores for each tumor region
        tc_dice = class_dice(outputs, Y_tensor, 2).item()
        ec_dice = class_dice(outputs, Y_tensor, 3).item()
        wt_dice = class_dice(outputs, Y_tensor, 4).item()

    return tc_dice, ec_dice, wt_dice


def All_view(X_test, Y_test, model):
    """Visualize model predictions with PyTorch"""
    model.eval()  # Set model to evaluation mode

    for num in range(X_test.shape[0]):
        # Convert input to PyTorch tensor
        X_tensor = torch.from_numpy(X_test[num:num+1]).float().to(device)
        Y_tensor = torch.from_numpy(Y_test[num:num+1]).float().to(device)

        # Get prediction
        with torch.no_grad():
            pred_tensor = model(X_tensor)
            # Apply threshold
            preds = (pred_tensor > 0.2).float()

        # Move tensors to CPU for visualization
        X_cpu = X_tensor.cpu()
        Y_cpu = Y_tensor.cpu()
        preds_cpu = preds.cpu()

        # Arrange images for visualization
        GT, Pre, TC, EC, WT = arrange_img(X_cpu, Y_cpu, preds_cpu)

        # Calculate dice scores
        TC_s, EC_s, WT_s = evaluate_dice_scores(
            model, X_test[num:num+1], Y_test[num:num+1])

        # Create visualization
        fig, ax = plt.subplots(1, 5, figsize=(30, 15))

        # Convert tensors to numpy for matplotlib
        if isinstance(Pre, torch.Tensor):
            Pre = Pre.numpy()
        if isinstance(GT, torch.Tensor):
            GT = GT.numpy()
        if isinstance(TC, torch.Tensor):
            TC = TC.numpy()
        if isinstance(EC, torch.Tensor):
            EC = EC.numpy()
        if isinstance(WT, torch.Tensor):
            WT = WT.numpy()

        ax[0].imshow(GT)
        ax[0].set_title(f'GT : {num}', fontsize=25)
        ax[0].axis("off")

        ax[1].imshow(Pre)
        ax[1].set_title('Prediction', fontsize=25)
        ax[1].axis("off")

        ax[2].imshow(TC)
        ax[2].set_title(f'TC : {TC_s:.2f}', fontsize=25)
        ax[2].axis("off")

        ax[3].imshow(EC)
        ax[3].set_title(f'EC : {EC_s:.2f}', fontsize=25)
        ax[3].axis("off")

        ax[4].imshow(WT)
        ax[4].set_title(f'WT : {WT_s:.2f}', fontsize=25)
        ax[4].axis("off")

        plt.tight_layout()
        plt.show()
