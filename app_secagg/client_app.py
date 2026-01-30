"""
Flower ClientApp with real cryptographic Secure Aggregation (SecAgg+).
Uses secaggplus_mod so client updates are masked before sending.
Run via: flwr run .
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
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
        dp_noise_fraction=0.0,
        dp_noise_scale=0.01,
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
        self.dp_noise_fraction = dp_noise_fraction
        self.dp_noise_scale = dp_noise_scale
        self.net = None  # created in fit to match server's make_net()

    def _get_net(self):
        if self.net is None:
            from app_secagg.task import make_net
            self.net = make_net(num_classes=self.num_classes)
            # Immediately move model to device to avoid device mismatches
            self.net = self.net.to(self.device)
        return self.net

    def get_parameters(self, config):
        return get_weights(self._get_net())

    def set_parameters(self, parameters):
        set_weights(self._get_net(), parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Store initial weights for delta computation
        initial_weights = get_weights(self._get_net())
        
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
        
        # Get trained weights
        trained_weights = get_weights(net)
        
        # Apply DP noise to weight deltas if enabled
        if self.dp_noise_fraction > 0.0:
            noisy_weights = self._apply_dp_noise_to_deltas(initial_weights, trained_weights)
        else:
            noisy_weights = trained_weights
        
        return noisy_weights, len(self.trainloader.dataset), results
    
    def _apply_dp_noise_to_deltas(self, initial_weights, trained_weights):
        """
        Apply differential privacy noise to weight deltas.
        Adds Gaussian noise to a fraction of parameters in the weight deltas.
        """
        noisy_weights = []
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        
        for init_param, trained_param in zip(initial_weights, trained_weights):
            # Compute delta (weight update)
            delta = trained_param - init_param
            
            # Create a copy for noisy delta
            noisy_delta = delta.copy()
            
            # Select random fraction of parameters to add noise to
            total_params = delta.size
            num_noisy_params = int(total_params * self.dp_noise_fraction)
            
            if num_noisy_params > 0:
                # Flatten delta, add noise to random subset, then reshape
                flat_delta = delta.flatten()
                flat_noisy_delta = noisy_delta.flatten()
                
                # Randomly select indices to add noise to
                noisy_indices = rng.choice(
                    len(flat_delta), 
                    size=num_noisy_params, 
                    replace=False
                )
                
                # Add Gaussian noise to selected parameters
                noise = rng.normal(0, self.dp_noise_scale, size=num_noisy_params)
                flat_noisy_delta[noisy_indices] += noise
                
                # Reshape back to original shape
                noisy_delta = flat_noisy_delta.reshape(delta.shape)
            
            # Return initial weights + noisy delta
            noisy_weights.append(init_param + noisy_delta)
        
        return noisy_weights

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
    
    # DP noise parameters (default to 0.0 = disabled)
    dp_noise_fraction = float(run_config.get("dp-noise-fraction", 0.0))
    dp_noise_scale = float(run_config.get("dp-noise-scale", 0.01))

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
        dp_noise_fraction=dp_noise_fraction,
        dp_noise_scale=dp_noise_scale,
    ).to_client()


app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
