import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

class NumpyDataset(Dataset):
    """A simple Dataset wrapping numpy arrays X (features) and y (labels)."""
    def __init__(self, X, y):
        # Convert numpy arrays to torch Tensors
        self.X = torch.from_numpy(X).float()  # .float() or .long() depending on your data
        self.y = torch.from_numpy(y).float()  # or .long() if it's class labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NumpyWrapper(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=8, val_ratio=0.2):
        """
        Args:
            X (np.ndarray): Features of shape (N, ...)
            y (np.ndarray): Labels of shape (N, )
            batch_size (int): Batch size to use in the DataLoader
            val_ratio (float): Fraction of data to use for validation
        """
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.val_ratio = val_ratio

    def setup(self, stage=None):
        """
        Use setup to split data into train/val sets and
        create the corresponding datasets.
        """
        dataset = NumpyDataset(self.X, self.y)

        # Split dataset into train and val
        val_size = int(len(dataset) * self.val_ratio)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        # If you have a separate test set, you can create self.test_dataset here as well.

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
    def test_dataloader(self):
        # If you have a separate test set, return DataLoader for test dataset here
        return None
