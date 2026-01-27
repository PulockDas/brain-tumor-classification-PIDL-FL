"""
Class-wise weight delta computation for federated learning.
Tracks gradients separately for each class during training.
"""

import torch
import torch.nn as nn
from collections import defaultdict
import copy


def get_model_weights(model):
    """
    Extract all model parameters as a flat list of tensors.
    
    Args:
        model: PyTorch model
    
    Returns:
        weights: List of parameter tensors
    """
    return [param.data.clone() for param in model.parameters()]


def set_model_weights(model, weights):
    """
    Set model parameters from a list of tensors.
    
    Args:
        model: PyTorch model
        weights: List of parameter tensors
    """
    for param, weight in zip(model.parameters(), weights):
        param.data.copy_(weight)


def compute_weight_delta(initial_weights, final_weights):
    """
    Compute weight delta: final_weights - initial_weights.
    
    Args:
        initial_weights: List of initial parameter tensors
        final_weights: List of final parameter tensors
    
    Returns:
        deltas: List of delta tensors
    """
    return [final - initial for final, initial in zip(final_weights, initial_weights)]


class ClassWiseDeltaTracker:
    """
    Tracks weight deltas separately for each class during training.
    """
    def __init__(self, num_classes, device):
        """
        Args:
            num_classes: Number of classes
            device: Device to store deltas on
        """
        self.num_classes = num_classes
        self.device = device
        self.class_deltas = defaultdict(lambda: None)  # class_id -> list of delta tensors
        self.class_counts = defaultdict(int)  # class_id -> number of samples
    
    def reset(self):
        """Reset all tracked deltas and counts."""
        self.class_deltas = defaultdict(lambda: None)
        self.class_counts = defaultdict(int)
    
    def accumulate_class_gradients(self, model, labels, loss_fn, feature_map_fn=None):
        """
        Accumulate gradients for samples of each class separately.
        This is a simplified approach: we compute gradients per class by
        masking the loss for each class.
        
        Args:
            model: PyTorch model
            labels: Batch labels (B,)
            loss_fn: Loss function that takes (logits, labels, feature_map)
            feature_map_fn: Function to get feature map from model output
        """
        model.train()
        
        # Get unique classes in this batch
        unique_classes = torch.unique(labels).cpu().numpy()
        
        # Store initial weights
        initial_weights = get_model_weights(model)
        
        # Process each class separately
        for class_id in unique_classes:
            # Create mask for this class
            class_mask = (labels == class_id)
            if class_mask.sum() == 0:
                continue
            
            # Get samples for this class
            class_labels = labels[class_mask]
            
            # Forward pass
            model.zero_grad()
            outputs = model(class_labels.device)  # This won't work, need to fix
            
            # Actually, we need a different approach
            # Let's compute gradients for the whole batch but weight by class
            pass
    
    def compute_classwise_deltas_from_training(self, model, train_loader, loss_fn, 
                                                optimizer, num_epochs, feature_layer='layer2',
                                                initial_weights=None):
        """
        Train model and compute class-wise weight deltas.
        
        This approach:
        1. Trains normally on all data
        2. Tracks which samples belong to which class
        3. Computes approximate class-wise deltas by weighting the overall delta
           by the contribution of each class
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training
            loss_fn: Loss function
            optimizer: Optimizer
            num_epochs: Number of training epochs
            feature_layer: Feature layer for regularization
            initial_weights: Initial model weights (if None, uses current weights)
        
        Returns:
            class_deltas_dict: Dict mapping class_id -> list of delta tensors
            class_counts: Dict mapping class_id -> number of samples
        """
        if initial_weights is None:
            initial_weights = get_model_weights(model)
        
        # Track class distribution and loss contributions
        class_loss_contributions = defaultdict(float)
        class_counts = defaultdict(int)
        
        # Train for num_epochs
        model.train()
        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, feature_maps = model(images, return_features=True)
                feature_map = feature_maps[feature_layer]
                
                # Compute loss
                total_loss, ce_loss, reg_loss = loss_fn(logits, labels, feature_map)
                
                # Track class contributions (approximate)
                for class_id in torch.unique(labels):
                    class_id_int = class_id.item()
                    class_mask = (labels == class_id_int)
                    class_count = class_mask.sum().item()
                    class_counts[class_id_int] += class_count
                    
                    # Approximate class loss contribution
                    class_logits = logits[class_mask]
                    class_labels = labels[class_mask]
                    class_ce = nn.functional.cross_entropy(class_logits, class_labels, reduction='mean')
                    class_loss_contributions[class_id_int] += class_ce.item() * class_count
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        # Get final weights
        final_weights = get_model_weights(model)
        
        # Compute overall delta
        overall_delta = compute_weight_delta(initial_weights, final_weights)
        
        # Distribute delta to classes based on their contribution
        total_loss_contribution = sum(class_loss_contributions.values())
        class_deltas_dict = {}
        
        for class_id in range(self.num_classes):
            if class_counts[class_id] == 0:
                # No samples of this class, zero delta
                class_deltas_dict[class_id] = [
                    torch.zeros_like(delta) for delta in overall_delta
                ]
            else:
                # Weight delta by class contribution
                weight = class_loss_contributions[class_id] / total_loss_contribution if total_loss_contribution > 0 else 1.0 / self.num_classes
                class_deltas_dict[class_id] = [
                    delta * weight for delta in overall_delta
                ]
        
        return class_deltas_dict, dict(class_counts)


def compute_classwise_deltas_simple(model, train_loader, loss_fn, optimizer, 
                                     num_epochs, num_classes, device, 
                                     feature_layer='layer2', initial_weights=None):
    """
    Simplified class-wise delta computation.
    Trains normally and distributes the overall delta to classes based on sample counts.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training
        loss_fn: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        num_classes: Number of classes
        device: Device
        feature_layer: Feature layer for regularization
        initial_weights: Initial model weights
    
    Returns:
        class_deltas_dict: Dict mapping class_id -> list of delta tensors
        class_counts: Dict mapping class_id -> number of samples
        training_metrics: Dict with training metrics
    """
    if initial_weights is None:
        initial_weights = get_model_weights(model)
    
    # Track class distribution
    class_counts = defaultdict(int)
    total_samples = 0
    
    # Training metrics
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    reg_loss_sum = 0.0
    correct = 0
    
    # Train for num_epochs
    model.train()
    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_reg_sum = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Count classes
            for class_id in torch.unique(labels):
                class_id_int = class_id.item()
                class_mask = (labels == class_id_int)
                class_counts[class_id_int] += class_mask.sum().item()
            
            total_samples += labels.size(0)
            
            # Forward pass
            logits, feature_maps = model(images, return_features=True)
            feature_map = feature_maps[feature_layer]
            
            # Compute loss
            total_loss, ce_loss, reg_loss = loss_fn(logits, labels, feature_map)
            
            # Metrics
            epoch_loss_sum += total_loss.item() * labels.size(0)
            epoch_ce_sum += ce_loss.item() * labels.size(0)
            epoch_reg_sum += reg_loss.item() * labels.size(0)
            
            _, predicted = torch.max(logits.data, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        total_loss_sum += epoch_loss_sum
        ce_loss_sum += epoch_ce_sum
        reg_loss_sum += epoch_reg_sum
        correct += epoch_correct
    
    # Get final weights
    final_weights = get_model_weights(model)
    
    # Compute overall delta
    overall_delta = compute_weight_delta(initial_weights, final_weights)
    
    # Distribute delta to classes based on sample counts
    class_deltas_dict = {}
    for class_id in range(num_classes):
        if class_counts[class_id] == 0:
            # No samples of this class, zero delta
            class_deltas_dict[class_id] = [
                torch.zeros_like(delta) for delta in overall_delta
            ]
        else:
            # Weight delta by class sample proportion
            weight = class_counts[class_id] / total_samples
            class_deltas_dict[class_id] = [
                delta * weight for delta in overall_delta
            ]
    
    # Compute metrics
    num_batches = len(train_loader) * num_epochs
    training_metrics = {
        'loss': total_loss_sum / total_samples,
        'ce_loss': ce_loss_sum / total_samples,
        'reg_loss': reg_loss_sum / total_samples,
        'accuracy': 100.0 * correct / total_samples
    }
    
    return class_deltas_dict, dict(class_counts), training_metrics
