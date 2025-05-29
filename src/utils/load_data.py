import numpy as np
import os
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import os


class BrainDataset(Dataset):
    def __init__(self, data_dir, subset='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        #  take the subset of data
        # Normalize path separators for cross-platform compatibility
        x_path = os.path.join("data\X" + subset + ".npy")
        y_path = os.path.join("data\Y" + subset + ".npy")
        # y_path = os.path.join(data_dir, f"Y{subset}.npy")

        try:
            #  take the subset of data
            self.X = np.load(x_path, mmap_mode='r')
            self.Y = np.load(y_path, mmap_mode='r')
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print(f"Attempted to load files from: {x_path} and {y_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Check if data directory exists: {os.path.exists(data_dir)}")
            raise


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)

        # Convert to PyTorch's NCHW format
        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y).permute(2, 0, 1)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y


def get_data_loaders(data_dir, batch_size=8, train_val_split=0.8, seed=42):

    full_train_dataset = BrainDataset(data_dir)

    # Calculate sizes for train and validation splits
    train_size = int(train_val_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Use the existing validation set as the test seta
    test_dataset = BrainDataset(data_dir, "val")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size
    )

    return train_loader, val_loader, test_loader