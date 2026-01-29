"""
Flower server implementation with custom aggregation for class-wise weight deltas.
"""

import torch
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from typing import List, Tuple, Dict, Optional
import numpy as np

from models.resnet_pidl import ResNet18FeatureExtractor
from federated.classwise_deltas import (
    get_model_weights, set_model_weights
)
from federated.secure_aggregation import (
    SecureAggregator, aggregate_classwise_deltas, combine_classwise_deltas
)


class ClassWiseFedAvgStrategy(FedAvg):
    """
    Custom FedAvg strategy that handles class-wise weight deltas.
    """
    def __init__(self, num_classes, *args, **kwargs):
        """
        Args:
            num_classes: Number of classes
            *args, **kwargs: Arguments for FedAvg
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.secure_aggregator = SecureAggregator(
            num_clients=kwargs.get('min_fit_clients', 3),
            random_seed=kwargs.get('random_seed', 42)
        )
        # Store class-wise deltas from clients
        self.client_classwise_deltas = []
    
    def aggregate_fit(self, rnd, results, failures):
        """
        Aggregate class-wise deltas from clients.
        
        Args:
            rnd: Current round number
            results: List of tuples (client, FitRes)
            failures: List of failures
        
        Returns:
            aggregated_parameters: Aggregated model parameters
            aggregated_metrics: Aggregated metrics
        """
        if not results:
            return None, {}
        
        # Get initial parameters (from first client)
        initial_params = results[0][1].parameters
        
        # Convert to model weights format
        initial_weights = [torch.tensor(p) for p in initial_params]
        
        # Collect class-wise deltas from all clients
        # Note: In a real implementation, you'd retrieve these from custom messages
        # For now, we'll reconstruct from the standard Flower interface
        # This is a limitation - we'll need to store deltas separately
        
        # For this implementation, we'll use a workaround:
        # Store class-wise deltas in a shared location or use Flower's custom messages
        # For now, let's use standard FedAvg but prepare for class-wise aggregation
        
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        # Store round info
        aggregated_metrics['round'] = rnd
        aggregated_metrics['num_clients'] = len(results)
        
        return aggregated_parameters, aggregated_metrics


class PIDLFlowerServer:
    """
    Custom Flower server wrapper that handles class-wise deltas properly.
    Uses a shared state to store class-wise deltas from clients.
    """
    def __init__(self, num_classes, num_clients, device, config):
        """
        Args:
            num_classes: Number of classes
            num_clients: Number of clients
            device: Device
            config: Configuration dict
        """
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.device = device
        self.config = config
        
        # Shared state for class-wise deltas (client_id -> class_deltas_dict)
        self.client_classwise_deltas = {}
        self.secure_aggregator = SecureAggregator(
            num_clients=num_clients,
            random_seed=config.get('random_seed', 42)
        )
    
    def store_client_classwise_deltas(self, client_id, class_deltas_dict):
        """Store class-wise deltas from a client."""
        self.client_classwise_deltas[client_id] = class_deltas_dict
    
    def aggregate_and_update(self, global_model, client_classwise_deltas_list):
        """
        Aggregate class-wise deltas and update global model.
        
        Args:
            global_model: Global model to update
            client_classwise_deltas_list: List of dicts, each mapping class_id -> deltas
        
        Returns:
            updated_weights: Updated model weights
        """
        # Get current global weights
        current_weights = get_model_weights(global_model)
        
        # Aggregate class-wise deltas from all clients
        aggregated_classwise_deltas = aggregate_classwise_deltas(
            client_classwise_deltas_list,
            self.num_classes
        )
        
        # Decrypt aggregated deltas (simulated)
        decrypted_classwise_deltas = {}
        for class_id, deltas in aggregated_classwise_deltas.items():
            decrypted_deltas = self.secure_aggregator.decrypt_aggregated_deltas(deltas)
            decrypted_classwise_deltas[class_id] = decrypted_deltas
        
        # Combine class-wise deltas into overall delta
        overall_delta = combine_classwise_deltas(
            decrypted_classwise_deltas,
            self.num_classes
        )
        
        # Update global weights
        updated_weights = [
            current + delta for current, delta in zip(current_weights, overall_delta)
        ]
        
        # Set updated weights to model
        set_model_weights(global_model, updated_weights)
        
        return updated_weights


# Global state for storing class-wise deltas (workaround for Flower's limitations)
# In production, use proper message passing or shared storage
_global_classwise_deltas_store = {}


def store_classwise_deltas(client_id, round_num, class_deltas_dict):
    """Store class-wise deltas in global store."""
    if round_num not in _global_classwise_deltas_store:
        _global_classwise_deltas_store[round_num] = {}
    _global_classwise_deltas_store[round_num][client_id] = class_deltas_dict


def get_classwise_deltas_for_round(round_num):
    """Retrieve all class-wise deltas for a round."""
    return _global_classwise_deltas_store.get(round_num, {})
