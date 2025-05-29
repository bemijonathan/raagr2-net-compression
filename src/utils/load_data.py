from email.headerregistry import DateHeader
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
        self.X = np.load(os.path.join(
            data_dir, f"X{subset}.npy"), mmap_mode='r')
        self.Y = np.load(os.path.join(
            data_dir, f"Y{subset}.npy"), mmap_mode='r')

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

    # Use the existing validation set as the test set
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
