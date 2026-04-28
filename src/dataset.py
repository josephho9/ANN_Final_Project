"""
PyTorch Dataset and DataLoader factories for the jumpshot keypoint sequences.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.preprocessing import augment_sequence


class JumpshotDataset(Dataset):
    """
    Dataset for basketball jumpshot form classification.

    Args:
        X: (N, T, input_dim) keypoint sequences
        y: (N,) integer class labels  (0 = poor form, 1 = good form)
        augment: apply random augmentation per sample
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]          # (T, input_dim)
        if self.augment:
            x = augment_sequence(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
):
    """
    Split data into train / val / test and return DataLoaders.
    Uses a WeightedRandomSampler on the training set to handle class imbalance.
    """
    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        random_state=seed,
        stratify=y,
    )
    val_ratio = config.VAL_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=seed,
        stratify=y_temp,
    )

    train_ds = JumpshotDataset(X_train, y_train, augment=True)
    val_ds = JumpshotDataset(X_val, y_val, augment=False)
    test_ds = JumpshotDataset(X_test, y_test, augment=False)

    # Weighted sampler for imbalanced classes
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts[y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, (X_test, y_test)
