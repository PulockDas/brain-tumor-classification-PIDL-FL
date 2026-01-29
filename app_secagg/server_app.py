"""
Flower ServerApp with real cryptographic Secure Aggregation (SecAgg+).
Run via: flwr run .
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from logging import DEBUG
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.logger import update_console_handler
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from app_secagg.logging_strategy import FedAvgWithLogging
from app_secagg.task import get_weights, make_net


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate client metrics by weighted average."""
    accuracies = [n * m.get("accuracy", 0.0) for n, m in metrics]
    examples = [n for n, _ in metrics]
    total = sum(examples)
    if total == 0:
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / total}


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_config = context.run_config
    is_demo = run_config.get("is-demo", False)
    num_classes = int(run_config.get("num-classes", 4))

    ndarrays = get_weights(make_net(num_classes=num_classes))
    parameters = ndarrays_to_parameters(ndarrays)

    log_dir = str(run_config.get("log-dir", "results"))
    data_root = str(run_config.get("data-root", "/content/drive/MyDrive/PhysNet/datasets/brain_tumor_mri"))
    config_dict = {k: v for k, v in run_config.items() if isinstance(v, (str, int, float, bool))}

    num_rounds = int(run_config.get("num-server-rounds", 5))
    
    strategy = FedAvgWithLogging(
        log_dir=log_dir,
        data_root=data_root,
        num_classes=num_classes,
        config=config_dict,
        regularizer_type=str(run_config.get("regularizer-type", "perona_malik")),
        lambda_pm=float(run_config.get("lambda-pm", 0.1)),
        k=float(run_config.get("k", 1.0)),
        feature_layer=str(run_config.get("feature-layer", "layer2")),
        num_rounds=num_rounds,
        fraction_fit=1.0,
        min_fit_clients=int(run_config.get("min-fit-clients", 3)),
        fraction_evaluate=0.0 if is_demo else float(run_config.get("fraction-evaluate", 0.0)),
        min_available_clients=int(run_config.get("min-fit-clients", 3)),
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if is_demo:
        update_console_handler(DEBUG, True, True)

    fit_workflow = SecAggPlusWorkflow(
        num_shares=int(run_config.get("num-shares", 3)),
        reconstruction_threshold=int(run_config.get("reconstruction-threshold", 2)),
        max_weight=int(run_config.get("max-weight", 2**20 - 1)),
    )
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, legacy_context)
