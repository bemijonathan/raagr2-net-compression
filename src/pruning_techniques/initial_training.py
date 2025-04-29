import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from architecture.model import DLUNet
from utils.custom_loss import Weighted_BCEnDice_loss
from architecture.model import mean_iou, class_dice
from utils.custom_metric import dice_coef
from utils.load_data import BrainDataset


# Define the directory containing the data
train_data = r"./data"
batch_size = 16
num_epochs = 5


# Initialize model, optimizer and loss
device = torch.device("mps")
print(f"Using device: {device}")
model = DLUNet(in_channels=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5, factor=0.1)
criterion = Weighted_BCEnDice_loss


train_dataset = BrainDataset(train_data)
val_dataset = BrainDataset(train_data, "val")

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


best_val_loss = float('inf')

checkpoint_interval = 1  # Save a checkpoint every 5 epochs
os.makedirs('model', exist_ok=True)

torch.save(model.state_dict(), 'model/dlu_net_model_best_training_2.pth')


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training phase
    model.train()
    train_loss = 0.0

    for step, (images, masks) in enumerate(train_loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = Weighted_BCEnDice_loss(outputs, masks)
        dice = dice_coef(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Log progress for training
        if step % 10 == 0 or step == len(train_loader):
            print(
                f"\r{step}/{len(train_loader)} [==============================] - loss: {loss.item():.4f}", end="")
            print(
                f"\r{step}/{len(train_loader)} [==============================] - metrics: {mean_iou(outputs, masks):.4f} c_2: {class_dice(outputs, masks, 2)} c_3: {class_dice(outputs, masks, 3)} c_4: {class_dice(outputs, masks, 4)}", end="")

    train_loss /= len(train_loader)
    print(f"\nTraining Loss: {train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for step, (images, masks) in enumerate(val_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = Weighted_BCEnDice_loss(outputs, masks)
            dice = dice_coef(outputs, masks)

            val_loss += loss.item()
            val_dice += dice.item()

            # Log progress for validation
            if step % 10 == 0 or step == len(val_loader):
                print(
                    f"\rValidation {step}/{len(val_loader)} [==============================]"
                    f" - val_loss: {loss.item():.4f}"
                    f" - val_dice: {dice.item():.4f}",
                    end=""
                )

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    print(
        f"\nValidation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")

    # Step the scheduler with validation loss
    scheduler.step(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'model/dlu_net_model_best.pth')
        print(f"Saved new best model with val_loss: {val_loss:.4f}")

    # Save an intermediate checkpoint every 'checkpoint_interval' epochs
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f"model/dlu_net_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Intermediate checkpoint saved at epoch {epoch+1} to '{checkpoint_path}'")

    print('-' * 60)
