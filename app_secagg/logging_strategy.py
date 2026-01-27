"""
Custom FedAvg strategy that logs round and client metrics to CSV for plotting.
Used by the SecAgg+ app so the Colab notebook can plot results after flwr run.
"""

from pathlib import Path
import sys

# Project root on path for utils
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from typing import Any, Dict, List, Optional, Tuple

import torch

from flwr.common import FitRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from app_secagg.task import (
    get_global_test_loader,
    get_weights,
    make_net,
    set_weights,
    test,
)


class FedAvgWithLogging(FedAvg):
    """
    FedAvg that after each round evaluates the global model on the global test set
    and logs round/client metrics to fl_rounds.csv and fl_clients.csv for plotting.
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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._data_root = data_root
        self._num_classes = num_classes
        self._regularizer_type = regularizer_type
        self._lambda_pm = lambda_pm
        self._k = k
        self._feature_layer = feature_layer
        self._test_loader = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._round_log_path = self._log_dir / "fl_rounds.csv"
        self._client_log_path = self._log_dir / "fl_clients.csv"
        self._round_headers_written = False
        self._client_headers_written = False

    def _ensure_test_loader(self):
        if self._test_loader is None:
            n = getattr(self, "min_fit_clients", 3)
            self._test_loader, _ = get_global_test_loader(
                self._data_root,
                batch_size=32,
                num_clients=n,
                num_workers=0,
                pin_memory=False,
            )
        if self._model is None:
            self._model = make_net(num_classes=self._num_classes)

    def _write_round_headers_if_needed(self):
        if self._round_headers_written:
            return
        with open(self._round_log_path, "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow([
                "round", "global_test_acc", "global_test_loss", "global_test_ce_loss",
                "global_test_reg_loss", "num_clients", "aggregation_time"
            ])
        self._round_headers_written = True

    def _write_client_headers_if_needed(self):
        if self._client_headers_written:
            return
        with open(self._client_log_path, "w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow([
                "round", "client_id", "train_loss", "train_acc", "train_ce_loss",
                "train_reg_loss", "num_samples", "class_distribution"
            ])
        self._client_headers_written = True

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, Exception]],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        # Log client metrics (fit results)
        self._write_client_headers_if_needed()
        import csv
        with open(self._client_log_path, "a", newline="") as f:
            w = csv.writer(f)
            for i, (proxy, fit_res) in enumerate(results):
                m = fit_res.metrics or {}
                w.writerow([
                    server_round - 1,
                    i,
                    m.get("val_loss", 0.0),
                    (m.get("accuracy", 0.0) or 0.0) * 100.0,
                    m.get("train_ce_loss", ""),
                    m.get("train_reg_loss", ""),
                    fit_res.num_examples,
                    "",
                ])

        # Evaluate global model on test set and log round metrics
        if agg_params is not None:
            self._ensure_test_loader()
            ndarrays = parameters_to_ndarrays(agg_params)
            set_weights(self._model, ndarrays)
            loss, acc = test(
                self._model,
                self._test_loader,
                self._device,
                num_classes=self._num_classes,
                regularizer_type=self._regularizer_type,
                lambda_pm=self._lambda_pm,
                k=self._k,
                feature_layer=self._feature_layer,
            )
            self._write_round_headers_if_needed()
            with open(self._round_log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    server_round - 1,
                    acc * 100.0,
                    loss,
                    loss,
                    0.0,
                    len(results) if results else 0,
                    0.0,
                ])

        return agg_params, agg_metrics
