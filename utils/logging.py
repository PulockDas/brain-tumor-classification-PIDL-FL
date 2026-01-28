"""
Logging utilities for federated learning experiments.
Saves minimal, plottable metrics to CSV and JSON (no heavy log dir).
Suitable for uploading fl_rounds.csv, fl_clients.csv, fl_eval.json, config.json,
and fl_summary.json to GitHub.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class FLLogger:
    """
    Logger for federated learning experiments.
    Writes only essential result files for plotting and later analysis.
    """

    def __init__(self, log_dir: str = 'results'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.round_log_path = self.log_dir / 'fl_rounds.csv'
        self.client_log_path = self.log_dir / 'fl_clients.csv'
        self.config_path = self.log_dir / 'config.json'
        self.eval_path = self.log_dir / 'fl_eval.json'
        self.summary_path = self.log_dir / 'fl_summary.json'

        self._eval_records: List[Dict[str, Any]] = []
        self._class_names: Optional[List[str]] = None
        self._init_csv_files()

    def _init_csv_files(self) -> None:
        round_headers = [
            'round', 'global_test_acc', 'global_test_loss', 'global_test_ce_loss',
            'global_test_reg_loss', 'num_clients', 'aggregation_time',
            'inference_time_sec', 'training_time_sec',
            'f1_macro', 'f1_micro', 'precision_macro', 'recall_macro',
        ]
        with open(self.round_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(round_headers)

        client_headers = [
            'round', 'client_id', 'train_loss', 'train_acc', 'train_ce_loss',
            'train_reg_loss', 'num_samples', 'train_time_sec', 'class_distribution',
        ]
        with open(self.client_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(client_headers)

    def log_round(
        self,
        round_num: int,
        global_test_metrics: Dict[str, Any],
        num_clients: int,
        aggregation_time: float = 0.0,
        training_time_sec: float = 0.0,
    ) -> None:
        row = [
            round_num,
            global_test_metrics.get('accuracy', 0.0),
            global_test_metrics.get('loss', 0.0),
            global_test_metrics.get('ce_loss', 0.0),
            global_test_metrics.get('reg_loss', 0.0),
            num_clients,
            aggregation_time,
            global_test_metrics.get('inference_time_sec', 0.0),
            training_time_sec,
            global_test_metrics.get('f1_macro', 0.0),
            global_test_metrics.get('f1_micro', 0.0),
            global_test_metrics.get('precision_macro', 0.0),
            global_test_metrics.get('recall_macro', 0.0),
        ]
        with open(self.round_log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        # Accumulate for fl_eval.json
        rec: Dict[str, Any] = {
            'round': round_num,
            'f1_macro': global_test_metrics.get('f1_macro', 0.0),
            'f1_micro': global_test_metrics.get('f1_micro', 0.0),
            'precision_macro': global_test_metrics.get('precision_macro', 0.0),
            'recall_macro': global_test_metrics.get('recall_macro', 0.0),
            'inference_time_sec': global_test_metrics.get('inference_time_sec', 0.0),
        }
        if 'confusion_matrix' in global_test_metrics:
            rec['confusion_matrix'] = global_test_metrics['confusion_matrix']
        if 'class_names' in global_test_metrics:
            self._class_names = global_test_metrics['class_names']
        self._eval_records.append(rec)

    def log_client(
        self,
        round_num: int,
        client_id: int,
        client_metrics: Dict[str, Any],
        class_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        class_dist_str = json.dumps(class_counts) if class_counts else '{}'
        row = [
            round_num,
            client_id,
            client_metrics.get('loss', 0.0),
            client_metrics.get('accuracy', 0.0),
            client_metrics.get('ce_loss', 0.0),
            client_metrics.get('reg_loss', 0.0),
            client_metrics.get('num_samples', 0),
            client_metrics.get('train_time_sec', 0.0),
            class_dist_str,
        ]
        with open(self.client_log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def save_config(self, config: Dict[str, Any]) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def finalize(self) -> Dict[str, Any]:
        """Write fl_eval.json and fl_summary.json; return summary dict."""
        eval_data: Dict[str, Any] = {
            'rounds': self._eval_records,
            'class_names': self._class_names,
        }
        with open(self.eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)

        summary = self.get_summary()
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary

    def load_rounds_df(self) -> pd.DataFrame:
        return pd.read_csv(self.round_log_path)

    def load_clients_df(self) -> pd.DataFrame:
        return pd.read_csv(self.client_log_path)

    def get_summary(self) -> Dict[str, Any]:
        rounds_df = self.load_rounds_df()
        clients_df = self.load_clients_df()

        n = len(rounds_df)
        summary: Dict[str, Any] = {
            'num_rounds': n,
            'num_clients': int(clients_df['client_id'].nunique()) if len(clients_df) else 0,
            'best_test_acc': float(rounds_df['global_test_acc'].max()) if n > 0 else 0.0,
            'final_test_acc': float(rounds_df['global_test_acc'].iloc[-1]) if n > 0 else 0.0,
            'final_test_loss': float(rounds_df['global_test_loss'].iloc[-1]) if n > 0 else 0.0,
            'best_f1_macro': float(rounds_df['f1_macro'].max()) if n > 0 and 'f1_macro' in rounds_df.columns else 0.0,
            'final_f1_macro': float(rounds_df['f1_macro'].iloc[-1]) if n > 0 and 'f1_macro' in rounds_df.columns else 0.0,
            'total_inference_time_sec': float(rounds_df['inference_time_sec'].sum()) if n > 0 and 'inference_time_sec' in rounds_df.columns else 0.0,
            'total_training_time_sec': float(rounds_df['training_time_sec'].sum()) if n > 0 and 'training_time_sec' in rounds_df.columns else 0.0,
        }
        return summary
