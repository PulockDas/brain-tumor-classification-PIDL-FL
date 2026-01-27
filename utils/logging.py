"""
Logging utilities for federated learning experiments.
Saves metrics to CSV and JSON for later plotting.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd


class FLLogger:
    """
    Logger for federated learning experiments.
    Saves metrics per round and per client.
    """
    def __init__(self, log_dir='results'):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log files
        self.round_log_path = self.log_dir / 'fl_rounds.csv'
        self.client_log_path = self.log_dir / 'fl_clients.csv'
        self.config_path = self.log_dir / 'config.json'
        
        # Initialize CSV files
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Round-level metrics
        round_headers = [
            'round', 'global_test_acc', 'global_test_loss', 'global_test_ce_loss',
            'global_test_reg_loss', 'num_clients', 'aggregation_time'
        ]
        
        with open(self.round_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(round_headers)
        
        # Client-level metrics
        client_headers = [
            'round', 'client_id', 'train_loss', 'train_acc', 'train_ce_loss',
            'train_reg_loss', 'num_samples', 'class_distribution'
        ]
        
        with open(self.client_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(client_headers)
    
    def log_round(self, round_num, global_test_metrics, num_clients, aggregation_time=0.0):
        """
        Log round-level metrics.
        
        Args:
            round_num: Round number
            global_test_metrics: Dict with test metrics
            num_clients: Number of clients in this round
            aggregation_time: Time taken for aggregation (seconds)
        """
        row = [
            round_num,
            global_test_metrics.get('accuracy', 0.0),
            global_test_metrics.get('loss', 0.0),
            global_test_metrics.get('ce_loss', 0.0),
            global_test_metrics.get('reg_loss', 0.0),
            num_clients,
            aggregation_time
        ]
        
        with open(self.round_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_client(self, round_num, client_id, client_metrics, class_counts=None):
        """
        Log client-level metrics.
        
        Args:
            round_num: Round number
            client_id: Client ID
            client_metrics: Dict with client training metrics
            class_counts: Dict mapping class_id -> count
        """
        class_dist_str = json.dumps(class_counts) if class_counts else '{}'
        
        row = [
            round_num,
            client_id,
            client_metrics.get('loss', 0.0),
            client_metrics.get('accuracy', 0.0),
            client_metrics.get('ce_loss', 0.0),
            client_metrics.get('reg_loss', 0.0),
            client_metrics.get('num_samples', 0),
            class_dist_str
        ]
        
        with open(self.client_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def save_config(self, config):
        """Save experiment configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_rounds_df(self):
        """Load round-level metrics as pandas DataFrame."""
        return pd.read_csv(self.round_log_path)
    
    def load_clients_df(self):
        """Load client-level metrics as pandas DataFrame."""
        return pd.read_csv(self.client_log_path)
    
    def get_summary(self):
        """Get summary statistics."""
        rounds_df = self.load_rounds_df()
        clients_df = self.load_clients_df()
        
        summary = {
            'num_rounds': len(rounds_df),
            'num_clients': clients_df['client_id'].nunique(),
            'best_test_acc': rounds_df['global_test_acc'].max() if len(rounds_df) > 0 else 0.0,
            'final_test_acc': rounds_df['global_test_acc'].iloc[-1] if len(rounds_df) > 0 else 0.0,
        }
        
        return summary
