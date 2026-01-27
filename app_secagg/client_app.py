"""
Flower ClientApp with real cryptographic Secure Aggregation (SecAgg+).
Uses secaggplus_mod so client updates are masked before sending.
Run via: flwr run .
"""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from app_secagg.task import get_weights, load_data, set_weights, test, train


class PIDLFlowerClient(NumPyClient):
    """Client that trains with PIDL loss and participates in SecAgg+."""

    def __init__(
        self,
        trainloader,
        valloader,
        num_classes,
        local_epochs,
        learning_rate,
        regularizer_type,
        lambda_pm,
        k,
        feature_layer,
        device,
    ):
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.regularizer_type = regularizer_type
        self.lambda_pm = lambda_pm
        self.k = k
        self.feature_layer = feature_layer
        self.device = device
        self.net = None  # created in fit to match server's make_net()

    def _get_net(self):
        if self.net is None:
            from app_secagg.task import make_net
            self.net = make_net(num_classes=self.num_classes)
        return self.net

    def get_parameters(self, config):
        return get_weights(self._get_net())

    def set_parameters(self, parameters):
        set_weights(self._get_net(), parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        net = self._get_net()
        results = train(
            net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            num_classes=self.num_classes,
            regularizer_type=self.regularizer_type,
            lambda_pm=self.lambda_pm,
            k=self.k,
            feature_layer=self.feature_layer,
        )
        return get_weights(net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(
            self._get_net(),
            self.valloader,
            self.device,
            num_classes=self.num_classes,
            regularizer_type=self.regularizer_type,
            lambda_pm=self.lambda_pm,
            k=self.k,
            feature_layer=self.feature_layer,
        )
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Build client for this partition using PIDL + SecAgg+."""
    node_config = context.node_config
    run_config = context.run_config
    partition_id = int(node_config.get("partition-id", 0))
    num_partitions = int(node_config.get("num-partitions", 3))
    data_root = run_config.get("data-root", "/content/drive/MyDrive/PhysNet/datasets/brain_tumor_mri")
    batch_size = int(run_config.get("batch-size", 32))
    local_epochs = int(run_config.get("local-epochs", 2))
    lr = float(run_config.get("learning-rate", 0.001))
    num_classes = int(run_config.get("num-classes", 4))
    regularizer_type = run_config.get("regularizer-type", "perona_malik")
    lambda_pm = float(run_config.get("lambda-pm", 0.1))
    k = float(run_config.get("k", 1.0))
    feature_layer = run_config.get("feature-layer", "layer2")

    trainloader, valloader, num_classes = load_data(
        partition_id,
        num_partitions,
        data_root,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return PIDLFlowerClient(
        trainloader,
        valloader,
        num_classes,
        local_epochs,
        lr,
        regularizer_type,
        lambda_pm,
        k,
        feature_layer,
        device,
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
