"""
Differential Privacy noise addition for federated learning.
Adds Gaussian noise to weight deltas before sending to server.
"""

import torch
import numpy as np


def add_dp_noise_to_deltas(deltas, noise_scale=0.01, noise_fraction=0.15, random_seed=None):
    """
    Add Gaussian differential privacy noise to a fraction of weight deltas.
    
    Args:
        deltas: List of delta tensors (weight deltas)
        noise_scale: Standard deviation of Gaussian noise (default: 0.01)
        noise_fraction: Fraction of parameters to add noise to (default: 0.15, i.e., 15%)
        random_seed: Random seed for reproducibility
    
    Returns:
        noisy_deltas: List of noisy delta tensors
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    noisy_deltas = []
    
    for delta in deltas:
        noisy_delta = delta.clone()
        
        # Flatten to compute total number of parameters
        flat_delta = delta.flatten()
        num_params = flat_delta.numel()
        num_noisy = int(num_params * noise_fraction)
        
        # Randomly select indices to add noise to
        if num_noisy > 0:
            indices = torch.randperm(num_params, device=delta.device)[:num_noisy]
            
            # Add Gaussian noise to selected parameters
            noise = torch.randn(num_noisy, device=delta.device) * noise_scale
            
            # Reshape and add noise
            flat_noisy = noisy_delta.flatten()
            flat_noisy[indices] += noise
            noisy_delta = flat_noisy.reshape(delta.shape)
        
        noisy_deltas.append(noisy_delta)
    
    return noisy_deltas


def add_dp_noise_to_classwise_deltas(class_deltas_dict, noise_scale=0.01, 
                                       noise_fraction=0.15, random_seed=None):
    """
    Add DP noise to class-wise weight deltas.
    
    Args:
        class_deltas_dict: Dict mapping class_id -> list of delta tensors
        noise_scale: Standard deviation of Gaussian noise
        noise_fraction: Fraction of parameters to add noise to
        random_seed: Random seed for reproducibility
    
    Returns:
        noisy_class_deltas_dict: Dict mapping class_id -> list of noisy delta tensors
    """
    noisy_class_deltas_dict = {}
    
    for class_id, deltas in class_deltas_dict.items():
        # Use class_id as part of seed for reproducibility
        seed = random_seed + class_id if random_seed is not None else None
        noisy_deltas = add_dp_noise_to_deltas(deltas, noise_scale, noise_fraction, seed)
        noisy_class_deltas_dict[class_id] = noisy_deltas
    
    return noisy_class_deltas_dict
