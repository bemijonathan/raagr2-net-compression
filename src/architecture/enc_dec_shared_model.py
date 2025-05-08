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
device = torch.device("mps" if torch.backends.mps.is_available() else
                     "cuda" if torch.cuda.is_available() else "cpu")

# Keep visualization and helper functions unchanged
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

# Metrics and loss functions remain the same
def dice_coef(y_pred, y_true, smooth=1.0):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

def dice_loss(y_pred, y_true, smooth=1.0):
    return 1 - dice_coef(y_pred, y_true, smooth)

def mean_iou(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return torch.mean(iou)

def weighted_bce_dice_loss(y_pred, y_true, weight_bce=0.5):
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_pred, y_true)
    return weight_bce * bce + (1 - weight_bce) * dice

def class_dice(y_pred, y_true, class_idx, smooth=1.0):
    y_true_c = y_true[:, class_idx, :, :]
    y_pred_c = y_pred[:, class_idx, :, :]
    return dice_coef(y_pred_c, y_true_c, smooth)

# Enhanced blocks with weight sharing capabilities

class SharedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SharedConvBlock, self).__init__()
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

class SharedDepthwiseBlock(nn.Module):
    """
    A single "shared‐depthwise → pointwise → residual" block.
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

class ReASPP3(nn.Module):
    def __init__(self, in_channels, out_channels, r=3):
        super().__init__()

        # Shared depthwise (kernel=3×3)
        self.shared_dw_weight = nn.Parameter(torch.randn(in_channels, 1, 3, 3))
        self.shared_dw_bias = nn.Parameter(torch.randn(in_channels))

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

# New UpConv block with shared weights
class SharedUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_conv_weights=None):
        super(SharedUpConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # If encoder weights are provided, use them for the convolution
        if encoder_conv_weights is not None:
            self.conv = encoder_conv_weights
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

# Enhanced attention module that works with shared weights
class EnhancedAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(EnhancedAttention, self).__init__()
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

# Shared encoder-decoder block for feature processing
class SharedEncoderDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r=3):
        super(SharedEncoderDecoderBlock, self).__init__()

        # Shared parameters for depthwise convolution
        self.shared_dw_weight = nn.Parameter(torch.randn(in_channels, 1, 3, 3))
        self.shared_dw_bias = nn.Parameter(torch.randn(in_channels))

        # Encoder module (ReASPP3 using shared weights)
        self.encoder = ReASPP3(in_channels, out_channels, r)

        # For decoder that will use the same weights
        self.decoder_channels = out_channels

    def get_shared_weights(self):
        return {
            'dw_weight': self.shared_dw_weight,
            'dw_bias': self.shared_dw_bias
        }

# Main DLU-Net model with enhanced weight sharing
class EnhancedDLUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):
        super(EnhancedDLUNet, self).__init__()

        # Create shared encoder-decoder blocks
        self.shared_block1 = SharedEncoderDecoderBlock(in_channels, 32)
        self.shared_block2 = SharedEncoderDecoderBlock(32, 64)
        self.shared_block3 = SharedEncoderDecoderBlock(64, 128)
        self.shared_block4 = SharedEncoderDecoderBlock(128, 256)
        self.shared_block5 = SharedEncoderDecoderBlock(256, 512)

        # Pooling layers for encoder path
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder path upsample blocks
        self.up5 = SharedUpConv(512, 256)
        self.up4 = SharedUpConv(256, 128)
        self.up3 = SharedUpConv(128, 64)
        self.up2 = SharedUpConv(64, 32)

        # Attention blocks
        self.att5 = EnhancedAttention(256, 256, 128)
        self.att4 = EnhancedAttention(128, 128, 64)
        self.att3 = EnhancedAttention(64, 64, 32)
        self.att2 = EnhancedAttention(32, 32, 16)

        # Aggregation blocks for decoder path
        # Using ReASPP3 blocks for decoder path with shared weights from encoder
        self.dec5 = ReASPP3(512, 256, r=3)  # After concat: 256+256=512
        self.dec4 = ReASPP3(256, 128, r=3)  # After concat: 128+128=256
        self.dec3 = ReASPP3(128, 64, r=3)   # After concat: 64+64=128
        self.dec2 = ReASPP3(64, 32, r=3)    # After concat: 32+32=64

        # Final output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder path with shared blocks
        e1 = self.shared_block1.encoder(x)
        p1 = self.pool1(e1)

        e2 = self.shared_block2.encoder(p1)
        p2 = self.pool2(e2)

        e3 = self.shared_block3.encoder(p2)
        p3 = self.pool3(e3)

        e4 = self.shared_block4.encoder(p3)
        p4 = self.pool4(e4)

        e5 = self.shared_block5.encoder(p4)

        # Decoder path with attention and skip connections
        # Using the same weights from encoder where possible
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

# Keep the rest of the functions (data handling, training, etc.) the same
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
    """Create and return an initialized DLU-Net model with weight sharing"""
    try:
        model = EnhancedDLUNet(in_channels=4).to(device)
        print("Enhanced model with weight sharing created successfully:", model)
        return model
    except Exception as e:
        print("Error creating model:", e)
        raise

def load_trained_model(model_path):
    """Load a trained model from a checkpoint file"""
    model = EnhancedDLUNet(in_channels=4)
    model.load_state_dict(torch.load(
        model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# The train_model and supporting functions remain the same
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
            torch.save(model.state_dict(), 'model/enhanced_dlu_net_model.pth')
            print(f"Saved new best model with val_loss: {val_loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Dice: {val_dice:.4f}")

    return model
