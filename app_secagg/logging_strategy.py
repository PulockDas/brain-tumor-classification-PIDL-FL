"""
Custom FedAvg strategy that logs round and client metrics to CSV/JSON for plotting.
Used by the SecAgg+ app so the Colab notebook can plot results after flwr run.
Writes fl_rounds.csv, fl_clients.csv, fl_eval.json, fl_summary.json, config.json.
"""

import csv
import json
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from flwr.common import FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from app_secagg.task import (
    evaluate_global,
    get_global_test_loader,
    get_weights,
    make_net,
    set_weights,
)


class FedAvgWithLogging(FedAvg):
    """
    FedAvg that after each round evaluates the global model on the global test set
    and logs round/client metrics plus fl_eval.json, fl_summary.json for plotting.
    """

    def __init__(
        self,
        log_dir: str,
        data_root: str,
        num_classes: int = 4,
        regularizer_type: str = "perona_malik",
        lambda_pm: float = 0.1,
        k: float = 1.0,
        feature_layer: str = "layer2",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            with open(self._log_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
        self._data_root = data_root
        self._num_classes = num_classes
        self._regularizer_type = regularizer_type
        self._lambda_pm = lambda_pm
        self._k = k
        self._feature_layer = feature_layer
        self._test_loader = None
        self._class_names = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._round_log_path = self._log_dir / "fl_rounds.csv"
        self._client_log_path = self._log_dir / "fl_clients.csv"
        self._eval_path = self._log_dir / "fl_eval.json"
        self._summary_path = self._log_dir / "fl_summary.json"
        self._round_headers_written = False
        self._client_headers_written = False
        self._eval_records: List[Dict[str, Any]] = []

    def _ensure_test_loader(self) -> None:
        if self._test_loader is None:
            n = getattr(self, "min_fit_clients", 3)
            self._test_loader, _, self._class_names = get_global_test_loader(
                self._data_root,
                batch_size=32,
                num_clients=n,
                num_workers=0,
                pin_memory=False,
            )
        if self._model is None:
            # Build server-side model once and place it on the chosen device.
            self._model = make_net(num_classes=self._num_classes).to(self._device)

    def _write_round_headers_if_needed(self) -> None:
        if self._round_headers_written:
            return
        with open(self._round_log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round", "global_test_acc", "global_test_loss", "global_test_ce_loss",
                "global_test_reg_loss", "num_clients", "aggregation_time",
                "inference_time_sec", "training_time_sec",
                "f1_macro", "f1_micro", "precision_macro", "recall_macro",
            ])
        self._round_headers_written = True

    def _write_client_headers_if_needed(self) -> None:
        if self._client_headers_written:
            return
        with open(self._client_log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "round", "client_id", "train_loss", "train_acc", "train_ce_loss",
                "train_reg_loss", "num_samples", "train_time_sec", "class_distribution",
            ])
        self._client_headers_written = True

    def _write_eval_and_summary(self) -> None:
        eval_data: Dict[str, Any] = {
            "rounds": self._eval_records,
            "class_names": self._class_names,
        }
        with open(self._eval_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        rounds_df = pd.read_csv(self._round_log_path)
        clients_df = pd.read_csv(self._client_log_path)
        n = len(rounds_df)
        summary: Dict[str, Any] = {
            "num_rounds": n,
            "num_clients": int(clients_df["client_id"].nunique()) if len(clients_df) else 0,
            "best_test_acc": float(rounds_df["global_test_acc"].max()) if n > 0 else 0.0,
            "final_test_acc": float(rounds_df["global_test_acc"].iloc[-1]) if n > 0 else 0.0,
            "final_test_loss": float(rounds_df["global_test_loss"].iloc[-1]) if n > 0 else 0.0,
            "best_f1_macro": float(rounds_df["f1_macro"].max()) if n > 0 and "f1_macro" in rounds_df.columns else 0.0,
            "final_f1_macro": float(rounds_df["f1_macro"].iloc[-1]) if n > 0 and "f1_macro" in rounds_df.columns else 0.0,
            "total_inference_time_sec": float(rounds_df["inference_time_sec"].sum()) if n > 0 and "inference_time_sec" in rounds_df.columns else 0.0,
            "total_training_time_sec": float(rounds_df["training_time_sec"].sum()) if n > 0 and "training_time_sec" in rounds_df.columns else 0.0,
        }
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, Exception]],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        self._write_client_headers_if_needed()
        training_time_sec = 0.0
        with open(self._client_log_path, "a", newline="") as f:
            w = csv.writer(f)
            for i, (proxy, fit_res) in enumerate(results):
                m = fit_res.metrics or {}
                acc = (m.get("accuracy", 0.0) or 0.0) * 100.0
                t_sec = m.get("train_time_sec", 0.0)
                training_time_sec += t_sec
                w.writerow([
                    server_round - 1,
                    i,
                    m.get("val_loss", 0.0),
                    acc,
                    "",
                    "",
                    fit_res.num_examples,
                    t_sec,
                    "{}",
                ])

        if agg_params is not None:
            self._ensure_test_loader()
            # Defensive: ensure server model is on the same device used for evaluation.
            # This avoids CPU-weight vs CUDA-input mismatches if the model was created
            # earlier (or deserialized) on a different device.
            self._model.to(self._device)
            ndarrays = parameters_to_ndarrays(agg_params)
            set_weights(self._model, ndarrays)
            metrics = evaluate_global(
                self._model,
                self._test_loader,
                self._device,
                num_classes=self._num_classes,
                class_names=self._class_names,
                regularizer_type=self._regularizer_type,
                lambda_pm=self._lambda_pm,
                k=self._k,
                feature_layer=self._feature_layer,
            )
            rec: Dict[str, Any] = {
                "round": server_round - 1,
                "f1_macro": metrics.get("f1_macro", 0.0),
                "f1_micro": metrics.get("f1_micro", 0.0),
                "precision_macro": metrics.get("precision_macro", 0.0),
                "recall_macro": metrics.get("recall_macro", 0.0),
                "inference_time_sec": metrics.get("inference_time_sec", 0.0),
            }
            if "confusion_matrix" in metrics:
                rec["confusion_matrix"] = metrics["confusion_matrix"]
            self._eval_records.append(rec)

            self._write_round_headers_if_needed()
            with open(self._round_log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    server_round - 1,
                    metrics.get("accuracy", 0.0),
                    metrics.get("loss", 0.0),
                    metrics.get("ce_loss", 0.0),
                    metrics.get("reg_loss", 0.0),
                    len(results) if results else 0,
                    0.0,
                    metrics.get("inference_time_sec", 0.0),
                    training_time_sec,
                    metrics.get("f1_macro", 0.0),
                    metrics.get("f1_micro", 0.0),
                    metrics.get("precision_macro", 0.0),
                    metrics.get("recall_macro", 0.0),
                ])
            self._write_eval_and_summary()

        return agg_params, agg_metrics
