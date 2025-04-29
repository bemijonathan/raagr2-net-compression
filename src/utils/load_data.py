import numpy as np
import os
import torch
from torch.utils.data import Dataset
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
