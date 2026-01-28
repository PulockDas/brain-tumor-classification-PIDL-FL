"""
Training utilities for federated learning evaluation and logging.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from models.resnet_pidl import ResNet18FeatureExtractor
from losses.pidl_loss import PIDLLoss

try:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def evaluate_model(
    model,
    test_loader,
    loss_fn,
    device,
    feature_layer='layer2',
    class_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict:
    """
    Evaluate model on test set.

    Returns metrics including loss, accuracy, confusion matrix, F1, precision,
    recall, and inference time (seconds).
    """
    # Ensure model and loss are on the same device as inputs
    model.to(device)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_reg_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits, feature_maps = model(images, return_features=True)
            feature_map = feature_maps[feature_layer]

            total_loss_batch, ce_loss_batch, reg_loss_batch = loss_fn(
                logits, labels, feature_map
            )
            total_loss += total_loss_batch.item() * labels.size(0)
            total_ce_loss += ce_loss_batch.item() * labels.size(0)
            total_reg_loss += reg_loss_batch.item() * labels.size(0)

            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    inference_time_sec = time.perf_counter() - t0

    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    metrics: Dict = {
        'loss': total_loss / total if total else 0.0,
        'ce_loss': total_ce_loss / total if total else 0.0,
        'reg_loss': total_reg_loss / total if total else 0.0,
        'accuracy': 100.0 * correct / total if total else 0.0,
        'inference_time_sec': inference_time_sec,
    }

    n_classes = num_classes
    if n_classes is None and all_labels:
        n_classes = int(max(all_labels)) + 1

    if _SKLEARN_AVAILABLE and total > 0 and n_classes is not None:
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        metrics['confusion_matrix'] = cm.tolist()

        # Handle edge case: not all classes may appear in y_true/y_pred
        try:
            metrics['f1_macro'] = float(
                f1_score(y_true, y_pred, average='macro', zero_division=0.0)
            )
            metrics['f1_micro'] = float(
                f1_score(y_true, y_pred, average='micro', zero_division=0.0)
            )
            metrics['precision_macro'] = float(
                precision_score(y_true, y_pred, average='macro', zero_division=0.0)
            )
            metrics['recall_macro'] = float(
                recall_score(y_true, y_pred, average='macro', zero_division=0.0)
            )
        except Exception:
            metrics['f1_macro'] = 0.0
            metrics['f1_micro'] = 0.0
            metrics['precision_macro'] = 0.0
            metrics['recall_macro'] = 0.0

        if class_names is not None and len(class_names) == n_classes:
            metrics['class_names'] = class_names

    return metrics
