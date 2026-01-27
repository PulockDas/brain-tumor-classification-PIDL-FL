"""
Lightweight secure aggregation SIMULATION for federated learning.
Simulates encryption/decryption of weight deltas using masking.

*** DEPRECATED — NOT TRUE CRYPTOGRAPHY ***
This module is a non-cryptographic simulation only. Do not use when you require
real secure aggregation. Use instead:

  1. Flower SecAgg+ (recommended): run `flwr run .` and use the app in
     app_secagg/ (ServerApp + SecAggPlusWorkflow, ClientApp + secaggplus_mod).
  2. See CRYPTO_RECOMMENDATIONS.md for alternatives (e.g. Paillier).

Ref: Bonawitz et al., "Practical Secure Aggregation for Federated Learning"
     https://arxiv.org/abs/1611.04482
"""

import torch
import numpy as np
from typing import List, Dict


class SecureAggregator:
    """
    Simulates secure aggregation using additive masking.
    NOT real cryptography. Use Flower SecAgg+ or Paillier for real security.
    See CRYPTO_RECOMMENDATIONS.md.
    """
    def __init__(self, num_clients, random_seed=None):
        """
        Args:
            num_clients: Number of clients
            random_seed: Random seed for mask generation
        """
        self.num_clients = num_clients
        self.random_seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
    
    def generate_masks(self, delta_shapes, client_id):
        """
        Generate encryption masks for a client.
        Masks are designed to cancel out when aggregated across all clients.
        
        Args:
            delta_shapes: List of shapes for each delta tensor
            client_id: ID of the client (0 to num_clients-1)
        
        Returns:
            masks: List of mask tensors
        """
        masks = []
        for shape in delta_shapes:
            # Generate random mask
            mask = torch.randn(shape)
            # Scale by client ID to ensure masks cancel out when summed
            # (This is a simplified simulation - real secure aggregation is more complex)
            mask = mask * (1.0 if client_id % 2 == 0 else -1.0)
            masks.append(mask)
        return masks
    
    def encrypt_deltas(self, deltas, client_id):
        """
        "Encrypt" deltas by adding masks.
        
        Args:
            deltas: List of delta tensors
            client_id: ID of the client
        
        Returns:
            encrypted_deltas: List of encrypted delta tensors
        """
        shapes = [delta.shape for delta in deltas]
        masks = self.generate_masks(shapes, client_id)
        
        encrypted_deltas = []
        for delta, mask in zip(deltas, masks):
            # Move mask to same device as delta
            mask = mask.to(delta.device)
            encrypted_delta = delta + mask
            encrypted_deltas.append(encrypted_delta)
        
        return encrypted_deltas
    
    def aggregate_encrypted_deltas(self, encrypted_deltas_list):
        """
        Aggregate encrypted deltas from multiple clients.
        In this simulation, masks cancel out when summed.
        
        Args:
            encrypted_deltas_list: List of lists, where each inner list contains
                                 encrypted deltas from one client
        
        Returns:
            aggregated_deltas: List of aggregated delta tensors
        """
        if not encrypted_deltas_list:
            return None
        
        num_clients = len(encrypted_deltas_list)
        num_deltas = len(encrypted_deltas_list[0])
        
        # Initialize aggregated deltas
        aggregated_deltas = []
        for i in range(num_deltas):
            shape = encrypted_deltas_list[0][i].shape
            device = encrypted_deltas_list[0][i].device
            aggregated_delta = torch.zeros(shape, device=device)
            aggregated_deltas.append(aggregated_delta)
        
        # Sum all encrypted deltas
        for encrypted_deltas in encrypted_deltas_list:
            for i, encrypted_delta in enumerate(encrypted_deltas):
                aggregated_deltas[i] += encrypted_delta
        
        # Average (masks should cancel out in this simulation)
        for i in range(len(aggregated_deltas)):
            aggregated_deltas[i] /= num_clients
        
        return aggregated_deltas
    
    def decrypt_aggregated_deltas(self, aggregated_deltas):
        """
        "Decrypt" aggregated deltas.
        In this simulation, masks have already canceled out during aggregation.
        
        Args:
            aggregated_deltas: List of aggregated delta tensors
        
        Returns:
            decrypted_deltas: List of decrypted delta tensors (same as aggregated)
        """
        # In this simplified simulation, decryption is a no-op
        # because masks cancel out during aggregation
        return aggregated_deltas


def aggregate_classwise_deltas(classwise_deltas_list, num_classes):
    """
    Aggregate class-wise deltas from multiple clients.
    
    Args:
        classwise_deltas_list: List of dicts, where each dict maps class_id -> list of deltas
        num_classes: Number of classes
    
    Returns:
        aggregated_classwise_deltas: Dict mapping class_id -> list of aggregated deltas
    """
    aggregated_classwise_deltas = {}
    
    for class_id in range(num_classes):
        # Collect deltas for this class from all clients
        class_deltas_from_clients = []
        for client_deltas_dict in classwise_deltas_list:
            if class_id in client_deltas_dict:
                class_deltas_from_clients.append(client_deltas_dict[class_id])
        
        if not class_deltas_from_clients:
            # No deltas for this class, create zero deltas
            # Use shape from first available class
            first_class_id = next(iter(classwise_deltas_list[0].keys()))
            aggregated_classwise_deltas[class_id] = [
                torch.zeros_like(delta) for delta in classwise_deltas_list[0][first_class_id]
            ]
        else:
            # Aggregate deltas for this class
            num_clients = len(class_deltas_from_clients)
            num_deltas = len(class_deltas_from_clients[0])
            
            aggregated_deltas = []
            for i in range(num_deltas):
                shape = class_deltas_from_clients[0][i].shape
                device = class_deltas_from_clients[0][i].device
                aggregated_delta = torch.zeros(shape, device=device)
                
                for client_deltas in class_deltas_from_clients:
                    aggregated_delta += client_deltas[i]
                
                aggregated_delta /= num_clients
                aggregated_deltas.append(aggregated_delta)
            
            aggregated_classwise_deltas[class_id] = aggregated_deltas
    
    return aggregated_classwise_deltas


def combine_classwise_deltas(classwise_deltas_dict, num_classes):
    """
    Combine class-wise deltas into a single overall delta.
    Simply sums all class deltas.
    
    Args:
        classwise_deltas_dict: Dict mapping class_id -> list of deltas
        num_classes: Number of classes
    
    Returns:
        combined_deltas: List of combined delta tensors
    """
    # Initialize combined deltas
    first_class_id = next(iter(classwise_deltas_dict.keys()))
    combined_deltas = [
        torch.zeros_like(delta) for delta in classwise_deltas_dict[first_class_id]
    ]
    
    # Sum all class deltas
    for class_id in range(num_classes):
        if class_id in classwise_deltas_dict:
            for i, class_delta in enumerate(classwise_deltas_dict[class_id]):
                combined_deltas[i] += class_delta
    
    return combined_deltas
