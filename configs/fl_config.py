"""
Configuration for federated learning experiments.
"""

import argparse
from typing import Dict


def get_default_fl_config() -> Dict:
    """Get default federated learning configuration."""
    return {
        # Data
        'data_root': '/content/drive/MyDrive/PhysNet/datasets/brain_tumor_mri',
        'num_clients': 3,
        'test_split': 0.15,
        'batch_size': 32,
        'image_size': 224,
        'augment': True,
        'num_workers': 4,
        'pin_memory': True,
        
        # Model
        'num_classes': 4,
        'pretrained': True,
        'feature_layer': 'layer2',
        
        # PIDL Loss
        'regularizer_type': 'perona_malik',  # 'perona_malik', 'isotropic', 'none'
        'lambda_pm': 0.1,
        'k': 1.0,
        
        # Federated Learning
        'num_rounds': 10,
        'local_epochs': 5,
        'learning_rate': 0.001,
        'optimizer': 'adam',  # 'adam' or 'sgd'
        'weight_decay': 1e-4,
        'momentum': 0.9,
        
        # Differential Privacy
        'dp_noise_scale': 0.01,
        'dp_noise_fraction': 0.15,  # Fraction of parameters to add noise to (10-20%)
        
        # Secure Aggregation
        'use_secure_aggregation': True,
        
        # Logging
        'log_dir': 'results',
        'save_checkpoints': True,
        'checkpoint_dir': 'checkpoints',
        
        # Random seed
        'random_seed': 42,
    }


def get_config_from_args() -> Dict:
    """Get configuration from command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Learning with PIDL')
    
    # Data arguments
    parser.add_argument('--data-root', type=str,
                        default='/content/drive/MyDrive/PhysNet/datasets/brain_tumor_mri',
                        help='Root directory containing Training folder')
    parser.add_argument('--num-clients', type=int, default=3,
                        help='Number of federated clients (default: 3)')
    parser.add_argument('--test-split', type=float, default=0.15,
                        help='Fraction of data for global test set (default: 0.15)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (default: 224)')
    
    # Model arguments
    parser.add_argument('--regularizer-type', type=str, default='perona_malik',
                        choices=['perona_malik', 'isotropic', 'none'],
                        help='Regularizer type (default: perona_malik)')
    parser.add_argument('--lambda-pm', type=float, default=0.1,
                        help='PIDL regularization weight (default: 0.1)')
    parser.add_argument('--k', type=float, default=1.0,
                        help='Conduction parameter for Perona-Malik (default: 1.0)')
    parser.add_argument('--feature-layer', type=str, default='layer2',
                        choices=['layer1', 'layer2', 'layer3', 'layer4'],
                        help='Feature layer for regularization (default: layer2)')
    
    # FL arguments
    parser.add_argument('--num-rounds', type=int, default=10,
                        help='Number of FL rounds (default: 10)')
    parser.add_argument('--local-epochs', type=int, default=5,
                        help='Number of local training epochs per round (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer (default: adam)')
    
    # DP arguments
    parser.add_argument('--dp-noise-scale', type=float, default=0.01,
                        help='DP noise scale (default: 0.01)')
    parser.add_argument('--dp-noise-fraction', type=float, default=0.15,
                        help='Fraction of parameters to add noise to (default: 0.15)')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='results',
                        help='Directory for logs (default: results)')
    
    # Random seed
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Convert to dict
    config = vars(args)
    
    # Add defaults for missing keys
    default_config = get_default_fl_config()
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    return config
