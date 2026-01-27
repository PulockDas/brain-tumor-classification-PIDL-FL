"""
Flower client implementation for federated learning with PIDL loss and class-wise deltas.
"""

import torch
import torch.optim as optim
from flwr.client import ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from typing import Dict, List, Tuple

from models.resnet_pidl import ResNet18FeatureExtractor
from losses.pidl_loss import PIDLLoss
from federated.classwise_deltas import (
    get_model_weights, set_model_weights, compute_classwise_deltas_simple
)
from federated.dp_noise import add_dp_noise_to_classwise_deltas
from federated.secure_aggregation import SecureAggregator


class PIDLFlowerClient(NumPyClient):
    """
    Flower client for PIDL-based federated learning with class-wise weight deltas.
    """
    def __init__(self, cid, train_loader, num_classes, device, config):
        """
        Args:
            cid: Client ID
            train_loader: DataLoader for this client's training data
            num_classes: Number of classes
            device: Device (cuda/cpu)
            config: Configuration dict with training hyperparameters
        """
        self.cid = cid
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.device = device
        self.config = config
        
        # Initialize model
        self.model = ResNet18FeatureExtractor(
            num_classes=num_classes,
            pretrained=config.get('pretrained', True)
        ).to(device)
        
        # Initialize loss function
        self.loss_fn = PIDLLoss(
            regularizer_type=config.get('regularizer_type', 'perona_malik'),
            k=config.get('k', 1.0),
            lambda_pm=config.get('lambda_pm', 0.1),
            num_classes=num_classes
        ).to(device)
        
        # Secure aggregator for encryption
        self.secure_aggregator = SecureAggregator(
            num_clients=config.get('num_clients', 3),
            random_seed=config.get('random_seed', 42) + cid
        )
        
        # Store initial weights (will be set by server)
        self.initial_weights = None
    
    def get_parameters(self, config):
        """Get current model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=False)
        
        # Store initial weights for delta computation (before training)
        self.initial_weights = get_model_weights(self.model)
    
    def fit(self, parameters, config):
        """
        Train model locally and return class-wise weight deltas.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dict
        
        Returns:
            parameters: Updated parameters (not used in our case)
            num_examples: Number of training examples
            metrics: Training metrics
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Training hyperparameters
        local_epochs = config.get('local_epochs', self.config.get('local_epochs', 5))
        learning_rate = config.get('learning_rate', self.config.get('learning_rate', 0.001))
        optimizer_type = config.get('optimizer', self.config.get('optimizer', 'adam'))
        
        # Create optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        
        # Train and compute class-wise deltas
        class_deltas_dict, class_counts, training_metrics = compute_classwise_deltas_simple(
            model=self.model,
            train_loader=self.train_loader,
            loss_fn=self.loss_fn,
            optimizer=optimizer,
            num_epochs=local_epochs,
            num_classes=self.num_classes,
            device=self.device,
            feature_layer=self.config.get('feature_layer', 'layer2'),
            initial_weights=self.initial_weights
        )
        
        # Add DP noise to class-wise deltas
        noise_scale = config.get('dp_noise_scale', self.config.get('dp_noise_scale', 0.01))
        noise_fraction = config.get('dp_noise_fraction', self.config.get('dp_noise_fraction', 0.15))
        
        noisy_class_deltas_dict = add_dp_noise_to_classwise_deltas(
            class_deltas_dict,
            noise_scale=noise_scale,
            noise_fraction=noise_fraction,
            random_seed=self.config.get('random_seed', 42) + self.cid
        )
        
        # Encrypt class-wise deltas (simulated)
        encrypted_class_deltas_dict = {}
        for class_id, deltas in noisy_class_deltas_dict.items():
            encrypted_deltas = self.secure_aggregator.encrypt_deltas(deltas, self.cid)
            encrypted_class_deltas_dict[class_id] = encrypted_deltas
        
        # Convert to numpy for Flower
        # Store encrypted deltas in a custom format
        # We'll need to return these through a custom mechanism
        # For now, we'll store them in the client and return via metrics
        
        # Get current parameters (for Flower compatibility)
        updated_params = self.get_parameters({})
        
        # Prepare metrics
        metrics = {
            'train_loss': training_metrics['loss'],
            'train_ce_loss': training_metrics['ce_loss'],
            'train_reg_loss': training_metrics['reg_loss'],
            'train_accuracy': training_metrics['accuracy'],
            'num_samples': sum(class_counts.values()),
            'class_counts': class_counts,
            # Store encrypted deltas in a way we can retrieve them
            # Note: This is a workaround - in practice, you'd use Flower's custom message passing
            'client_id': self.cid,
            'has_classwise_deltas': True
        }
        
        # Store encrypted deltas for server to retrieve
        # In a real implementation, you'd use Flower's custom message passing
        self.encrypted_class_deltas = encrypted_class_deltas_dict
        
        return updated_params, sum(class_counts.values()), metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluate model on local test data (if available).
        In our setup, evaluation is centralized on server.
        """
        # Set parameters
        self.set_parameters(parameters)
        
        # For now, return dummy metrics
        # In practice, clients might have local validation sets
        return 0.0, 1, {}


def create_client_fn(train_loaders, num_classes, device, config):
    """
    Create a function that returns a client instance.
    
    Args:
        train_loaders: List of DataLoaders, one per client
        num_classes: Number of classes
        device: Device
        config: Configuration dict
    
    Returns:
        client_fn: Function that takes cid and returns a NumPyClient
    """
    def client_fn(cid: str) -> NumPyClient:
        """Create a single client instance."""
        cid_int = int(cid)
        return PIDLFlowerClient(
            cid=cid_int,
            train_loader=train_loaders[cid_int],
            num_classes=num_classes,
            device=device,
            config=config
        )
    
    return client_fn
