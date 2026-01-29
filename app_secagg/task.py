"""
Task definitions for Flower SecAgg+ app: model, data, train, test.
Uses PIDL loss and ResNet-18 from this project.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path when running via flwr run
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.resnet_pidl import ResNet18FeatureExtractor
from losses.pidl_loss import PIDLLoss
from data.dataset_utils import create_fl_data_loaders
from federated.training_utils import evaluate_model


# Cache for partitioned data (keyed by data_root, num_partitions, ...)
_data_cache = None


def make_net(num_classes=4, pretrained=True, seed=42):
    """Build ResNet-18 PIDL model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return ResNet18FeatureExtractor(num_classes=num_classes, pretrained=pretrained)


def get_weights(net):
    """Extract model parameters as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Load model parameters from list of numpy arrays."""
    # Match dtype/device of the existing model parameters/buffers to avoid
    # CPU/GPU tensor mismatches during forward passes.
    ref_state = net.state_dict()
    # Get device from model (more reliable than individual params)
    # If model has no parameters yet, default to CPU (will be moved later)
    try:
        model_device = next(net.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    params_dict = zip(ref_state.keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.as_tensor(v, dtype=ref_state[k].dtype, device=model_device)
            for k, v in params_dict
        }
    )
    net.load_state_dict(state_dict, strict=True)


def load_data(
    partition_id: int,
    num_partitions: int,
    data_root: str,
    batch_size: int = 32,
    image_size: int = 224,
    augment: bool = True,
    test_split: float = 0.15,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_state: int = 42,
):
    """
    Load train/val loaders for the given partition.
    Uses stratified brain-tumor FL partitioning; test set is reused as val for client-side eval.
    """
    global _data_cache
    cache_key = (data_root, num_partitions, test_split, random_state)
    if _data_cache is None or _data_cache.get("key") != cache_key:
        client_loaders, test_loader, num_classes, _ = create_fl_data_loaders(
            data_root=data_root,
            num_clients=num_partitions,
            test_split=test_split,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            image_size=image_size,
            augment=augment,
            random_state=random_state,
        )
        _data_cache = {
            "key": cache_key,
            "client_loaders": client_loaders,
            "test_loader": test_loader,
            "num_classes": num_classes,
        }
    if partition_id >= len(_data_cache["client_loaders"]):
        raise ValueError(
            f"partition_id {partition_id} >= num_partitions {num_partitions}"
        )
    train_loader = _data_cache["client_loaders"][partition_id]
    val_loader = _data_cache["test_loader"]
    num_classes = _data_cache["num_classes"]
    return train_loader, val_loader, num_classes


def get_global_test_loader(
    data_root: str,
    batch_size: int = 32,
    test_split: float = 0.15,
    num_clients: int = 3,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = False,
    random_state: int = 42,
):
    """Return (test_loader, num_classes, class_names) for server-side global evaluation."""
    _, test_loader, num_classes, class_names = create_fl_data_loaders(
        data_root=data_root,
        num_clients=num_clients,
        test_split=test_split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        image_size=image_size,
        augment=False,
        random_state=random_state,
    )
    return test_loader, num_classes, class_names


def train(
    net,
    trainloader,
    valloader,
    epochs,
    learning_rate,
    device,
    num_classes=4,
    regularizer_type="perona_malik",
    lambda_pm=0.1,
    k=1.0,
    feature_layer="layer2",
) -> Dict[str, Any]:
    """Train model with PIDL loss. Returns metrics dict (val_loss, accuracy, train_time_sec)."""
    # Ensure model is on the correct device before training
    # This is critical to avoid device mismatches (CPU vs GPU)
    net = net.to(device)
    loss_fn = PIDLLoss(
        regularizer_type=regularizer_type,
        k=k,
        lambda_pm=lambda_pm,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    net.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            logits, feature_maps = net(images, return_features=True)
            feat = feature_maps[feature_layer]
            total_loss, _, _ = loss_fn(logits, labels, feat)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    train_time_sec = time.perf_counter() - t0
    loss, acc = test(net, valloader, device, num_classes, regularizer_type, lambda_pm, k, feature_layer)
    return {"val_loss": loss, "accuracy": acc, "train_time_sec": train_time_sec}


def test(
    net,
    testloader,
    device,
    num_classes=4,
    regularizer_type="perona_malik",
    lambda_pm=0.1,
    k=1.0,
    feature_layer="layer2",
) -> Tuple[float, float]:
    """Evaluate model; returns (loss, accuracy)."""
    # Ensure model is on the correct device before evaluation
    # This is critical to avoid device mismatches (CPU vs GPU)
    net = net.to(device)
    loss_fn = PIDLLoss(
        regularizer_type=regularizer_type,
        k=k,
        lambda_pm=lambda_pm,
        num_classes=num_classes,
    ).to(device)
    net.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits, feature_maps = net(images, return_features=True)
            feat = feature_maps[feature_layer]
            l, _, _ = loss_fn(logits, labels, feat)
            loss_sum += l.item() * labels.size(0)
            _, pred = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    loss = loss_sum / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return loss, accuracy


def evaluate_global(
    net,
    testloader,
    device,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
    regularizer_type: str = "perona_malik",
    lambda_pm: float = 0.1,
    k: float = 1.0,
    feature_layer: str = "layer2",
) -> Dict[str, Any]:
    """Full evaluation (loss, accuracy, confusion matrix, F1, inference time)."""
    # Ensure model is on the correct device before evaluation
    # This is critical to avoid device mismatches (CPU vs GPU)
    net = net.to(device)
    loss_fn = PIDLLoss(
        regularizer_type=regularizer_type,
        k=k,
        lambda_pm=lambda_pm,
        num_classes=num_classes,
    ).to(device)
    return evaluate_model(
        net,
        testloader,
        loss_fn,
        device,
        feature_layer=feature_layer,
        class_names=class_names,
        num_classes=num_classes,
    )
