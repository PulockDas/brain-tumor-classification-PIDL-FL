# Federated Learning for Brain Tumor Classification with PIDL

This repository extends the [Brain-Tumor-Classification-with-PM-Regulizer](https://github.com/PulockDas/Brain-Tumor-Classification-with-PM-Regulizer) project to support federated learning with class-wise weight deltas, differential privacy, and secure aggregation.

## Features

- **Federated Learning**: Multi-client FL simulation using Flower framework
- **Physics-Informed Deep Learning (PIDL)**: Perona-Malik anisotropic diffusion + isotropic diffusion regularizers
- **Class-wise Weight Deltas**: Separate weight updates per class (glioma, meningioma, no_tumor, pituitary)
- **Differential Privacy**: Gaussian noise added to 10-20% of weight parameters
- **Secure Aggregation**: Real cryptography via Flower SecAgg+ (recommended), or Paillier; see [CRYPTO_RECOMMENDATIONS.md](CRYPTO_RECOMMENDATIONS.md).
- **Stratified Data Partitioning**: Ensures balanced class distribution across clients
- **Comprehensive Logging**: CSV/JSON logs for plotting and analysis

## Dataset

The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle.

Expected directory structure:
```
brain_tumor_mri/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── Testing/  (optional, not used in FL)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Recommended: Colab notebook (full pipeline with true encryption)

Running **`notebooks/01_brain_mri_FL.ipynb`** in Google Colab is the main way to run the full project with **real cryptographic secure aggregation** (Flower SecAgg+):

1. Mount Drive, clone this repo (or upload it), then run **all cells** in order.
2. The notebook installs the project, runs `flwr run .` with SecAgg+, and writes `fl_rounds.csv` and `fl_clients.csv` under `log-dir`.
3. The last cells plot global test accuracy, client accuracies, and losses from those CSVs.

Set `DATA_ROOT` to your data path (e.g. `/content/drive/MyDrive/.../brain_tumor_mri`). Use **`LOG_DIR = "/content/results"`** (or `./results`) to avoid storing logs on Drive; copy the result files to the repo later if you want to commit them (see **Result files and GitHub** below).

### Local run with SecAgg+ (no Colab)

```bash
pip install -e .
flwr run . --run-config "data-root=/path/to/brain_tumor_mri" "log-dir=results" "num-server-rounds=10" "local-epochs=5"
```

The SecAgg+ app lives in `app_secagg/`. See **[CRYPTO_RECOMMENDATIONS.md](CRYPTO_RECOMMENDATIONS.md)** for other crypto options (e.g. Paillier).

### Optional: Custom loop (class-wise deltas + DP, **no** true crypto)

```bash
python train_fl.py --data-root /path/to/brain_tumor_mri --num-clients 3 --num-rounds 10 --local-epochs 5 --log-dir results
```

This path uses the **deprecated simulated** secure aggregation in `federated/secure_aggregation.py` and is **not** cryptographically secure. Prefer the notebook or `flwr run .` when you need real encryption.

## Configuration

Key configuration parameters:

- `--num-clients`: Number of federated clients (default: 3, max: 5)
- `--test-split`: Fraction of data for global test set (default: 0.15)
- `--num-rounds`: Number of FL rounds (default: 10)
- `--local-epochs`: Local training epochs per round (default: 5)
- `--regularizer-type`: `perona_malik`, `isotropic`, or `none`
- `--lambda-pm`: PIDL regularization weight (default: 0.1)
- `--dp-noise-scale`: DP noise standard deviation (default: 0.01)
- `--dp-noise-fraction`: Fraction of parameters to add noise to (default: 0.15)

## Federated Learning Process (SecAgg+ path — used by the notebook and `flwr run .`)

1. **Data partitioning**: 15–20% of data is split as a global test set (never used in training); the rest is stratified across N client partitions.
2. **Each FL round**: Clients receive the same global weights, train locally with PIDL loss, then send **masked** updates. The server runs the **SecAgg+** protocol (key share, collect masked vectors, unmask) so it sees only the **aggregated** update. The global model is updated and evaluated on the held-out test set; round/client metrics are written to `log-dir` for plotting.
3. **Inference**: Only the global server model is evaluated (centralized).

## Results and Logging

All runs write **minimal, plottable** result files under `log-dir` (e.g. `LOG_DIR` in the notebook or `--log-dir results`):

| `fl_rounds.csv` | Per round: global test acc/loss, CE/reg loss, **inference_time_sec**, **training_time_sec**, **f1_macro**, **f1_micro**, **precision_macro**, **recall_macro**, num_clients, aggregation_time |
| `fl_clients.csv` | Per client per round: train loss/acc, CE/reg loss, num_samples, **train_time_sec**, class_distribution |
| `fl_eval.json` | Per-round **confusion matrices**, class names, F1/precision/recall, inference time (for plotting) |
| `fl_summary.json` | Best/final test acc, best/final F1, total inference/training time, num_rounds, etc. |
| `config.json` | Experiment configuration |

Use a **local** `log-dir` (e.g. `./results` or `/content/results` on Colab) to avoid filling Drive. Disable checkpoints via config (`save_checkpoints: false`) to save disk space. The notebook plots from these CSVs. **GitHub:** `.gitignore` excludes `results/*` except the five files above; add and commit only those to share results.

## Project Structure

```
.
├── app_secagg/                  # SecAgg+ app (true crypto) — used by notebook and flwr run
│   ├── server_app.py            # ServerApp + SecAggPlusWorkflow + FedAvgWithLogging
│   ├── client_app.py            # ClientApp + secaggplus_mod
│   ├── task.py                  # Model, data, train, test
│   └── logging_strategy.py      # Writes fl_rounds.csv, fl_clients.csv
├── models/
│   └── resnet_pidl.py           # ResNet-18 with feature extraction
├── losses/
│   └── pidl_loss.py            # PIDL loss (Perona-Malik + isotropic)
├── data/
│   └── dataset_utils.py         # Data loading and stratified partitioning
├── federated/                   # Used by train_fl.py (optional custom loop)
│   ├── classwise_deltas.py
│   ├── dp_noise.py
│   ├── secure_aggregation.py    # Deprecated simulation — do not use for real crypto
│   ├── flower_client.py
│   ├── flower_server.py
│   └── training_utils.py
├── utils/
│   └── logging.py
├── configs/
│   └── fl_config.py
├── notebooks/
│   └── 01_brain_mri_FL.ipynb   # Main Colab notebook (runs flwr run with SecAgg+)
├── train_fl.py                  # Optional custom loop (no true crypto)
├── pyproject.toml               # flwr app entrypoints and config
├── requirements.txt
├── CRYPTO_RECOMMENDATIONS.md     # Why SecAgg+ and how to use Paillier
└── README.md
```

## Class-wise Weight Deltas

The system computes separate weight deltas for each class:
- Tracks which samples belong to which class during training
- Distributes overall weight delta to classes based on sample counts
- Server aggregates class-wise deltas separately, then combines them

This approach allows for more fine-grained updates and better handling of class imbalance.

## Citation

If you use this code, please cite:

1. The original PIDL work:
```
Perona, P., & Malik, J. (1990). Scale-space and edge detection using anisotropic diffusion. 
IEEE Transactions on pattern analysis and machine intelligence, 12(7), 629-639.
```

2. The original repository:
```
https://github.com/PulockDas/Brain-Tumor-Classification-with-PM-Regulizer
```

## License

[Add your license here]

## Future Work

- Add support for colon cancer dataset
- Implement more sophisticated secure aggregation
- Add more DP mechanisms (e.g., clipping)
- Support for more clients and heterogeneous data distributions
