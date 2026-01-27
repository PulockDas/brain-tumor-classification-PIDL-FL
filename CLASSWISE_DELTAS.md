# Class-wise Weight Deltas Implementation

## Overview

This document explains how class-wise weight deltas are implemented in the federated learning system.

## Concept

Instead of sending a single overall weight delta from each client, we compute and send **separate weight deltas for each class** (glioma, meningioma, no_tumor, pituitary). This allows the server to:

1. Aggregate updates per class separately
2. Better handle class imbalance
3. Apply different aggregation strategies per class if needed

## Implementation Details

### 1. Client-side: Computing Class-wise Deltas

**Location**: `federated/classwise_deltas.py` → `compute_classwise_deltas_simple()`

**Process**:
1. **Store initial weights** before training (from global model)
2. **Train locally** for E epochs with PIDL loss
3. **Track class distribution**: Count how many samples of each class were seen during training
4. **Compute overall delta**: `delta = final_weights - initial_weights`
5. **Distribute delta to classes**: Weight the overall delta by each class's sample proportion
   ```python
   class_delta[class_id] = overall_delta * (class_samples[class_id] / total_samples)
   ```

**Example**:
- Client sees 100 glioma, 50 meningioma, 30 no_tumor, 20 pituitary samples
- Total: 200 samples
- Overall delta is distributed as:
  - glioma_delta = overall_delta * (100/200) = 0.5 * overall_delta
  - meningioma_delta = overall_delta * (50/200) = 0.25 * overall_delta
  - no_tumor_delta = overall_delta * (30/200) = 0.15 * overall_delta
  - pituitary_delta = overall_delta * (20/200) = 0.1 * overall_delta

### 2. Adding DP Noise

**Location**: `federated/dp_noise.py` → `add_dp_noise_to_classwise_deltas()`

- Adds Gaussian noise to 10-20% of parameters in each class delta
- Noise scale is configurable (default: 0.01)
- Each class gets independent noise

### 3. Encryption (Simulated Secure Aggregation)

**Location**: `federated/secure_aggregation.py` → `SecureAggregator`

- Each client "encrypts" class-wise deltas by adding masks
- Masks are designed to cancel out when aggregated across all clients
- This is a **simplified simulation** - real secure aggregation uses cryptographic techniques

### 4. Server-side: Aggregating Class-wise Deltas

**Location**: `federated/secure_aggregation.py` → `aggregate_classwise_deltas()`

**Process**:
1. **Collect class-wise deltas** from all clients
2. **Aggregate per class**: Average deltas for each class separately
   ```python
   aggregated_delta[class_id] = mean(all_client_deltas[class_id])
   ```
3. **Decrypt aggregated deltas** (masks cancel out)
4. **Combine class deltas**: Sum all class deltas to get final update
   ```python
   final_delta = sum(aggregated_delta[class_id] for all classes)
   ```
5. **Update global model**: `global_weights = global_weights + final_delta`

## Why This Approach?

### Advantages:
1. **Class-aware updates**: Server knows which classes contributed to which updates
2. **Better handling of imbalance**: Can weight class contributions differently if needed
3. **More interpretable**: Can analyze which classes are learning faster
4. **Flexible aggregation**: Can implement class-specific aggregation strategies

### Limitations:
1. **Approximation**: We distribute overall delta by sample counts, not true per-class gradients
2. **Computational overhead**: Slightly more memory/storage for class-wise deltas
3. **Simplified secure aggregation**: Real crypto would be more complex

## Alternative Approaches (Not Implemented)

### Option 1: True Per-class Gradients
- Compute gradients separately for each class during backward pass
- More accurate but computationally expensive
- Requires modifying the training loop significantly

### Option 2: Weighted Aggregation by Class
- Send overall delta but include class distribution
- Server weights aggregation by class distribution
- Simpler but less fine-grained

## Current Implementation Status

✅ **Implemented**:
- Class-wise delta computation (sample-count based)
- DP noise per class
- Secure aggregation simulation
- Server-side class-wise aggregation
- Logging of class distributions

⚠️ **Simplifications**:
- Delta distribution is based on sample counts, not true per-class gradients
- Secure aggregation is simulated (not cryptographic)
- All classes use same aggregation strategy

## Future Improvements

1. **True per-class gradients**: Modify training loop to track gradients per class
2. **Class-specific aggregation**: Different strategies per class (e.g., weighted by class importance)
3. **Real secure aggregation**: Use cryptographic libraries (e.g., PySyft, TenSEAL)
4. **Class imbalance handling**: Weight class contributions by inverse frequency

## Usage Example

```python
# Client computes class-wise deltas
class_deltas_dict = {
    0: [delta_tensors_for_glioma],
    1: [delta_tensors_for_meningioma],
    2: [delta_tensors_for_no_tumor],
    3: [delta_tensors_for_pituitary]
}

# Server aggregates
aggregated_classwise = aggregate_classwise_deltas(
    [client1_deltas, client2_deltas, client3_deltas],
    num_classes=4
)

# Combine and update
final_delta = combine_classwise_deltas(aggregated_classwise, num_classes=4)
update_global_model(final_delta)
```

## References

- Original PIDL implementation: [Brain-Tumor-Classification-with-PM-Regulizer](https://github.com/PulockDas/Brain-Tumor-Classification-with-PM-Regulizer)
- Flower framework: [Flower Documentation](https://flower.dev/)
