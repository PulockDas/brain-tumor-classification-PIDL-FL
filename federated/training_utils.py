"""
Training utilities for federated learning evaluation and logging.
"""

import torch
from typing import Dict, List
from models.resnet_pidl import ResNet18FeatureExtractor
from losses.pidl_loss import PIDLLoss


def evaluate_model(model, test_loader, loss_fn, device, feature_layer='layer2'):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        loss_fn: Loss function
        device: Device
        feature_layer: Feature layer for regularization
    
    Returns:
        metrics: Dict with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_reg_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, feature_maps = model(images, return_features=True)
            feature_map = feature_maps[feature_layer]
            
            # Compute loss
            total_loss_batch, ce_loss_batch, reg_loss_batch = loss_fn(
                logits, labels, feature_map
            )
            
            total_loss += total_loss_batch.item() * labels.size(0)
            total_ce_loss += ce_loss_batch.item() * labels.size(0)
            total_reg_loss += reg_loss_batch.item() * labels.size(0)
            
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    metrics = {
        'loss': total_loss / total,
        'ce_loss': total_ce_loss / total,
        'reg_loss': total_reg_loss / total,
        'accuracy': 100.0 * correct / total
    }
    
    return metrics
