"""
Data loading and stratified FL partitioning for the Brain Tumor MRI dataset.
Expects: data_root/Training/{glioma,meningioma,no_tumor,pituitary}/
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

# Default class names (alphabetical order matches ImageFolder)
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = 4


def _to_rgb_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Ensure 3 channels for ResNet (duplicate if grayscale)."""
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x


def _get_transforms(
    image_size: int,
    augment: bool,
) -> Tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(_to_rgb_if_needed),
        transforms.Normalize(mean=mean, std=std),
    ]
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        *base,
    ]) if augment else transforms.Compose(base)
    test_tf = transforms.Compose(base)
    return train_tf, test_tf


def _stratified_client_partition(
    indices: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    random_state: int,
) -> List[np.ndarray]:
    """Partition indices into num_clients subsets with stratified class distribution."""
    rng = np.random.RandomState(random_state)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    lab = np.asarray(labels)
    for c in np.unique(lab):
        pos = np.where(lab == c)[0]
        # pos indexes into labels array; we need indices[pos]
        idx = indices[pos]
        rng.shuffle(idx)
        for i, k in enumerate(idx):
            client_indices[i % num_clients].append(int(k))
    return [np.array(x) for x in client_indices]


def create_fl_data_loaders(
    data_root: str,
    num_clients: int = 3,
    test_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    augment: bool = True,
    random_state: int = 42,
) -> Tuple[List[DataLoader], DataLoader, int, List[str]]:
    """
    Create federated train loaders (one per client) and a global test loader.
    Uses stratified split for test and stratified partitioning across clients.

    Returns:
        client_loaders: List of DataLoaders, one per client.
        test_loader: DataLoader for the global test set.
        num_classes: Number of classes (4).
        class_names: Class names in index order.
    """
    root = Path(data_root)
    train_dir = root / "Training"
    if not train_dir.is_dir():
        train_dir = root
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Data root must contain 'Training' with glioma, meningioma, no_tumor, pituitary. Not found: {train_dir}"
        )

    train_tf, test_tf = _get_transforms(image_size, augment)
    full = ImageFolder(str(train_dir), transform=None)
    num_classes = len(full.classes)
    class_names = full.classes  # alphabetical

    all_indices = np.arange(len(full))
    labels = np.array([full.targets[i] for i in all_indices])

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_split,
        stratify=labels,
        random_state=random_state,
    )
    client_idx_list = _stratified_client_partition(
        train_idx, labels[train_idx], num_clients, random_state
    )

    # Build datasets with transforms. Use separate train/test transforms.
    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    test_ds = ImageFolder(str(train_dir), transform=test_tf)

    client_loaders = []
    for cidx in client_idx_list:
        subset = Subset(train_ds, cidx.tolist())
        client_loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
        )
    test_loader = DataLoader(
        Subset(test_ds, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return client_loaders, test_loader, num_classes, list(class_names)
