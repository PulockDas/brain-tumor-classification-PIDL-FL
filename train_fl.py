"""
Main script for federated learning training with PIDL loss and class-wise deltas.
"""

import torch
import flwr as fl
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from flwr.client import ClientApp, NumPyClient
import time
import os

from data.dataset_utils import create_fl_data_loaders
from models.resnet_pidl import ResNet18FeatureExtractor
from losses.pidl_loss import PIDLLoss
from federated.flower_client import PIDLFlowerClient, create_client_fn
from federated.flower_server import (
    PIDLFlowerServer, store_classwise_deltas, get_classwise_deltas_for_round
)
from federated.training_utils import evaluate_model
from utils.logging import FLLogger
from configs.fl_config import get_default_fl_config, get_config_from_args


def main():
    """Main training function."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    args, unknown = parser.parse_known_args()
    
    # Get configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_config_from_args()
    
    # Set random seed
    torch.manual_seed(config['random_seed'])
    import numpy as np
    np.random.seed(config['random_seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loaders
    print("Loading and partitioning data...")
    client_loaders, test_loader, num_classes, class_names = create_fl_data_loaders(
        data_root=config['data_root'],
        num_clients=config['num_clients'],
        test_split=config['test_split'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        image_size=config['image_size'],
        augment=config['augment'],
        random_state=config['random_seed']
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Number of clients: {len(client_loaders)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    for i, loader in enumerate(client_loaders):
        print(f"  Client {i} training samples: {len(loader.dataset)}")
    
    # Initialize logger
    logger = FLLogger(log_dir=config['log_dir'])
    logger.save_config(config)
    
    # Initialize global model
    global_model = ResNet18FeatureExtractor(
        num_classes=num_classes,
        pretrained=config['pretrained']
    ).to(device)
    
    # Initialize loss function
    loss_fn = PIDLLoss(
        regularizer_type=config['regularizer_type'],
        k=config['k'],
        lambda_pm=config['lambda_pm'],
        num_classes=num_classes
    ).to(device)
    
    # Initialize server
    fl_server = PIDLFlowerServer(
        num_classes=num_classes,
        num_clients=config['num_clients'],
        device=device,
        config=config
    )
    
    # Create client function
    client_fn = create_client_fn(client_loaders, num_classes, device, config)
    
    # Custom strategy that handles class-wise deltas
    strategy = FedAvg(
        min_fit_clients=config['num_clients'],
        min_available_clients=config['num_clients'],
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # Centralized evaluation
    )
    
    # Start Flower server
    print("\nStarting federated learning...")
    print(f"Number of rounds: {config['num_rounds']}")
    print(f"Local epochs per round: {config['local_epochs']}")
    
    # We'll use a custom approach to handle class-wise deltas
    # Store them in a shared location and aggregate manually
    
    # Initialize clients
    clients = []
    for cid in range(config['num_clients']):
        client = client_fn(str(cid))
        clients.append(client)
    
    # Federated learning loop
    for round_num in range(config['num_rounds']):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{config['num_rounds']}")
        print(f"{'='*60}")
        
        round_start_time = time.time()
        
        # Get current global parameters
        global_params = [
            val.cpu().numpy() for _, val in global_model.state_dict().items()
        ]
        
        # Train each client
        client_results = []
        client_classwise_deltas_list = []
        
        for cid, client in enumerate(clients):
            print(f"\nTraining Client {cid}...")
            
            # Set global parameters
            client.set_parameters(global_params)
            
            # Store initial weights in client for delta computation
            from federated.classwise_deltas import get_model_weights
            client.initial_weights = get_model_weights(client.model)
            
            # Train client
            fit_config = {
                'local_epochs': config['local_epochs'],
                'learning_rate': config['learning_rate'],
                'optimizer': config['optimizer'],
                'dp_noise_scale': config['dp_noise_scale'],
                'dp_noise_fraction': config['dp_noise_fraction'],
            }
            
            updated_params, num_samples, metrics = client.fit(global_params, fit_config)
            
            # Get class-wise deltas from client (stored internally)
            if hasattr(client, 'encrypted_class_deltas'):
                client_classwise_deltas_list.append(client.encrypted_class_deltas)
            else:
                # Fallback: compute deltas from updated params
                print(f"  Warning: Client {cid} did not store class-wise deltas, using standard aggregation")
            
            client_results.append((cid, updated_params, num_samples, metrics))
            
            # Log client metrics
            class_counts = metrics.get('class_counts', {})
            logger.log_client(
                round_num=round_num,
                client_id=cid,
                client_metrics={
                    'loss': metrics.get('train_loss', 0.0),
                    'accuracy': metrics.get('train_accuracy', 0.0),
                    'ce_loss': metrics.get('train_ce_loss', 0.0),
                    'reg_loss': metrics.get('train_reg_loss', 0.0),
                    'num_samples': num_samples,
                    'train_time_sec': metrics.get('train_time_sec', 0.0),
                },
                class_counts=class_counts,
            )
            t_s = metrics.get('train_time_sec', 0.0)
            print(f"  Client {cid} - Loss: {metrics.get('train_loss', 0.0):.4f}, "
                  f"Acc: {metrics.get('train_accuracy', 0.0):.2f}%, "
                  f"train: {t_s:.1f}s")
        
        # Aggregate class-wise deltas and update global model
        if client_classwise_deltas_list and len(client_classwise_deltas_list) > 0:
            print("\nAggregating class-wise deltas...")
            try:
                updated_weights = fl_server.aggregate_and_update(
                    global_model, client_classwise_deltas_list
                )
                print("Global model updated with class-wise aggregation.")
            except Exception as e:
                print(f"  Error in class-wise aggregation: {e}")
                print("  Falling back to standard FedAvg aggregation...")
                # Fallback to standard aggregation
                from federated.classwise_deltas import get_model_weights, set_model_weights, compute_weight_delta
                current_weights = get_model_weights(global_model)
                # Simple average of client updates
                avg_delta = None
                for cid, updated_params, _, _ in client_results:
                    updated_weights_torch = [torch.tensor(p) for p in updated_params]
                    delta = compute_weight_delta(current_weights, updated_weights_torch)
                    if avg_delta is None:
                        avg_delta = delta
                    else:
                        avg_delta = [a + d for a, d in zip(avg_delta, delta)]
                if avg_delta:
                    avg_delta = [d / len(client_results) for d in avg_delta]
                    updated_weights = [c + d for c, d in zip(current_weights, avg_delta)]
                    set_model_weights(global_model, updated_weights)
        else:
            # Fallback to standard aggregation if no class-wise deltas
            print("\nUsing standard FedAvg aggregation (no class-wise deltas available)...")
            from federated.classwise_deltas import get_model_weights, set_model_weights
            from federated.classwise_deltas import compute_weight_delta
            current_weights = get_model_weights(global_model)
            # Simple average of client updates
            avg_delta = None
            for cid, updated_params, _, _ in client_results:
                updated_weights_torch = [torch.tensor(p) for p in updated_params]
                delta = compute_weight_delta(current_weights, updated_weights_torch)
                if avg_delta is None:
                    avg_delta = delta
                else:
                    avg_delta = [a + d for a, d in zip(avg_delta, delta)]
            if avg_delta:
                avg_delta = [d / len(client_results) for d in avg_delta]
                updated_weights = [c + d for c, d in zip(current_weights, avg_delta)]
                set_model_weights(global_model, updated_weights)
        
        # Evaluate global model on test set
        print("\nEvaluating global model on test set...")
        test_metrics = evaluate_model(
            global_model, test_loader, loss_fn, device,
            feature_layer=config['feature_layer'],
            class_names=class_names,
            num_classes=num_classes,
        )
        inf_s = test_metrics.get('inference_time_sec', 0.0)
        f1 = test_metrics.get('f1_macro', 0.0)
        print(f"Test - Loss: {test_metrics['loss']:.4f}, "
              f"CE: {test_metrics['ce_loss']:.4f}, "
              f"Reg: {test_metrics['reg_loss']:.4f}, "
              f"Acc: {test_metrics['accuracy']:.2f}%, "
              f"F1(macro): {f1:.4f}, "
              f"inference: {inf_s:.2f}s")
        
        # Sum client training times for this round
        training_time_sec = sum(
            m.get('train_time_sec', 0.0) for _, _, _, m in client_results
        )
        aggregation_time = time.time() - round_start_time
        logger.log_round(
            round_num=round_num,
            global_test_metrics=test_metrics,
            num_clients=len(clients),
            aggregation_time=aggregation_time,
            training_time_sec=training_time_sec,
        )
        
        # Save checkpoint
        if config.get('save_checkpoints', True):
            checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'model_round_{round_num}.pth')
            torch.save({
                'round': round_num,
                'model_state_dict': global_model.state_dict(),
                'test_metrics': test_metrics,
            }, checkpoint_path)
    
    # Final summary and write fl_eval.json, fl_summary.json
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    summary = logger.finalize()
    print(f"Total rounds: {summary['num_rounds']}")
    print(f"Best test accuracy: {summary['best_test_acc']:.2f}%")
    print(f"Final test accuracy: {summary['final_test_acc']:.2f}%")
    print(f"Best F1 (macro): {summary.get('best_f1_macro', 0):.4f}")
    print(f"Total inference time: {summary.get('total_inference_time_sec', 0):.1f}s")
    print(f"Total training time: {summary.get('total_training_time_sec', 0):.1f}s")
    print(f"\nResults saved to: {config['log_dir']}")
    print("  (fl_rounds.csv, fl_clients.csv, fl_eval.json, config.json, fl_summary.json)")


if __name__ == '__main__':
    main()
