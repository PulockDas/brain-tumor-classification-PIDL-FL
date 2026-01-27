# Cryptographic Secure Aggregation — Recommendations

This document recommends **real cryptographic** options for secure aggregation in your federated learning pipeline. The previous implementation in `federated/secure_aggregation.py` was a **simulation** (not true encryption) and must not be used when you require genuine cryptographic security.

---

## 1. **Recommended: Flower SecAgg+ (Lightweight, Real Crypto)**

**Best choice** when you want real security without the cost of homomorphic encryption.

### What it is
- **SecAgg+** is Flower’s built-in secure aggregation protocol, based on the line of work from **Bonawitz et al.** [“Practical Secure Aggregation for Federated Learning on User-Held Data”](https://arxiv.org/abs/1611.04482).
- It uses **secret sharing** and **key agreement** (e.g. Diffie–Hellman), not homomorphic encryption.
- The server sees only the **sum of masked updates**; individual client updates stay private.

### Why it fits your setup
- **Lightweight**: No homomorphic encryption, so much cheaper than Paillier/FHE.
- **Real cryptography**: Proper secret-sharing and unmasking, not “add noise and hope it cancels.”
- **Already in your stack**: You use Flower + PyTorch; SecAgg+ is built into Flower.
- **Standard in FL**: Commonly cited and accepted in FL privacy literature.

### How to use it in this project
Use the **SecAgg+ app** provided in this repo:

1. **Run with real secure aggregation** (Flower SecAgg+):
   ```bash
   flwr run .
   ```
   (with a `pyproject.toml` that defines a ServerApp using `SecAggPlusWorkflow` and a ClientApp using `secaggplus_mod`).

2. **Reference implementation**: Flower’s official example:
   - Docs: [Secure aggregation with Flower (the SecAgg+ protocol)](https://flower.ai/docs/examples/flower-secure-aggregation.html)
   - Code: `https://github.com/adap/flower/tree/main/examples/flower-secure-aggregation`

### Flower SecAgg+ in short
- **Server**: Uses `SecAggPlusWorkflow` (or `SecAggWorkflow`) in the fit workflow.
- **Client**: Uses `secaggplus_mod` in the ClientApp `mods`.
- **Stages**: Setup → Share keys → Collect masked vectors → Unmask. The mod and workflow handle masking/unmasking; your training logic (e.g. PIDL, class-wise deltas) stays in `fit()` and still returns parameters. The mod masks them before send.

### Citation
- Bonawitz, K. et al. “Practical Secure Aggregation for Privacy-Preserving Machine Learning.” *CRYPTO 2017* / [arXiv:1611.04482](https://arxiv.org/abs/1611.04482).

---

## 2. **Alternative: Paillier (Additive Homomorphic Encryption)**

Use this if you **explicitly want “homomorphic encryption”** in the narrative (e.g. for the thesis), and can accept more computation and complexity than SecAgg+.

### What it is
- **Paillier** is **additively homomorphic**: you can add encrypted numbers and multiply by scalars without decrypting.
- Clients encrypt gradient/parameter updates with the server’s public key; the server sums ciphertexts and decrypts **only the aggregate**.

### Pros and cons
- **Pros**: Real public-key homomorphic encryption; server never sees plaintext updates; easy to describe as “homomorphic encryption” in the paper.
- **Cons**: Heavier than SecAgg+ (encryption/decryption, larger ciphertexts); you need to quantize or scale floats to integers; key and ciphertext management is on you.

### Libraries
- **`phe`** (Python Paillier): `pip install phe` — simple API, fine for prototyping.
- **`python-paillier`** (Data61): [python-paillier](https://github.com/data61/python-paillier) — more options, used in several FL papers.

### Typical usage pattern
1. Server generates Paillier keypair, sends public key to clients.
2. Each client encodes (e.g. quantizes) its update, encrypts with the public key, sends ciphertexts.
3. Server homomorphically adds ciphertexts, then decrypts once to get the aggregate update.
4. Server updates the global model with the decrypted aggregate.

### Example projects
- [fedavg_encrypt](https://github.com/heroding77/fedavg_encrypt): FedAvg + Paillier + differential privacy.
- [Paillier_federated_learning](https://github.com/BostonCrayfish/Paillier_federated_learning): FL with Paillier.

---

## 3. **What to avoid for “real crypto”**

- **Homomorphic encryption** (CKKS, BFV, etc.) for **full** gradient updates: usually too heavy for your setting unless you target a specific “HE-based FL” contribution.
- **Your current simulation** in `federated/secure_aggregation.py`: additive random masks that “cancel” when summed. That is **not** cryptographic secure aggregation and must not be used when you require a true cryptographic solution.

---

## 4. **Summary**

| Option            | Type                  | Cost        | Use when                                                                 |
|-------------------|-----------------------|------------|---------------------------------------------------------------------------|
| **Flower SecAgg+** | Secret-sharing / MPC  | Low        | You want real crypto, minimal extra cost, and to stay within Flower.     |
| **Paillier**       | Additive HE           | Moderate   | You want to say “homomorphic encryption” and can afford the overhead.    |
| **Simulation**     | Non-crypto masking    | —          | **Do not use** when you need a cryptographic solution.                   |

**Recommendation:** Use **Flower SecAgg+** as the default cryptographic solution in this project. Use **Paillier** only if your research narrative specifically requires homomorphic encryption.

This project provides an **app using Flower SecAgg+** (`app_secagg/` and `pyproject.toml`). For the custom loop in `train_fl.py`, migrate from the simulated secure aggregation to either (a) the SecAgg+ app invoked via `flwr run .`, or (b) a dedicated Paillier-based aggregation path implemented alongside your existing training code.
