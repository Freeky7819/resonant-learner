# ❓ Frequently Asked Questions (FAQ)

**Common questions about RCA and early stopping**

---

## General Questions

### What is RCA?

**Resonant Convergence Analysis (RCA)** is an intelligent early stopping system that detects when your model has truly converged by analyzing oscillation patterns in validation loss.

Unlike simple patience-based methods, RCA uses **resonance metrics** (β and ω) to distinguish between:
- 🔵 **Temporary plateaus** (keep training)
- 🟢 **True convergence** (stop training)

**Result:** Save 25-47% compute while maintaining quality.

---

### Why use RCA instead of simple early stopping?

**Traditional early stopping:**
```python
if val_loss hasn't improved for N epochs:
    stop()
```
**Problems:**
- Can stop too early (temporary plateau)
- Can wait too long (wasted compute)
- No learning rate adaptation

**RCA:**
```python
if val_loss plateau detected (β > 0.70) AND no improvement for N epochs:
    reduce learning rate
    if still no improvement:
        stop()
```
**Benefits:**
- ✅ Distinguishes temporary vs real plateaus
- ✅ Adaptive LR helps escape plateaus
- ✅ More reliable stopping decisions
- ✅ Better final model quality

**Real data:** RCA saves 36% compute on average across 4 datasets.

---

### What are β and ω?

**Beta (β) - Resonance Amplitude**
- Measures how stable validation loss is
- Range: 0 to 1
- **High β (>0.7):** Very stable, converged
- **Low β (<0.5):** Still oscillating, learning

**Omega (ω) - Resonance Frequency**
- Measures oscillation frequency
- Range: 0 to 12
- **ω ≈ 6.0:** Optimal training dynamics
- **ω → 0:** Very slow changes (plateau)

**You don't need to understand the math** - RCA uses these automatically to make smart stopping decisions.

---

### How much compute does RCA save?

**Real production results (NVIDIA L40S GPU):**

| Dataset | Baseline | RCA | Saved | Quality |
|---------|----------|-----|-------|---------|
| MNIST | 30 epochs | 18 epochs | **40%** | +0.12% ✅ |
| Fashion-MNIST | 30 epochs | 16 epochs | **47%** | -0.67% ✅ |
| CIFAR-10 | 60 epochs | 45 epochs | **25%** | +1.35% ✅ |
| BERT SST2 | 10 epochs | 7 epochs | **30%** | -0.11% ✅ |

**Average: 36% compute savings**

**CIFAR-10 bonus:** RCA achieved *better* accuracy by preventing overfitting!

---

### Does RCA hurt model quality?

**No!** Quality is maintained or improved:

- **Average accuracy delta:** +0.17% (slight improvement)
- **Statistical test:** p=0.71 (no significant difference)
- **CIFAR-10:** +1.35% improvement (overfitting prevention)

RCA loads the **best checkpoint** automatically, ensuring you get the best model, not the final model.

---

### What's the difference between Community and Pro editions?

**Quick comparison:**

| Feature | Community | Professional |
|---------|-----------|--------------|
| RCA Early Stopping | ✅ | ✅ Enhanced |
| Price | Free | Contact |
| Setup Time | ~1-2 hours | ~5 minutes |
| Auto-Configuration | ❌ | ✅ AutoCoach |
| SmartTeach | ❌ | ✅ Gradient modulation |

**Community:** Manual training loops, full control, free  
**Professional:** Zero-config, auto-optimization, includes support

See [full comparison](./EDITIONS_COMPARISON.md) for details.

Contact: zakelj.damjan@gmail.com

---

## Installation & Setup

### What are the requirements?

**Minimum:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (optional, GPU recommended)

**Installation:**
```bash
pip install torch torchvision
git clone https://github.com/...
cd resonant-convergence-analysis
pip install -e .
```

---

### Can I use RCA without GPU?

**Yes!** RCA works on CPU, just slower.

```python
device = torch.device('cpu')
model = model.to(device)
```

**Note:** Examples may take longer:
- MNIST: ~10 min (CPU) vs ~2 min (GPU)
- CIFAR-10: ~30 min (CPU) vs ~5 min (GPU)

---

### Does RCA work with my framework?

**Native support:**
- ✅ PyTorch (manual loops)
- ✅ Plain Python training scripts

**Possible with effort:**
- ⚠️ PyTorch Lightning (custom callback)
- ⚠️ HuggingFace Trainer (manual integration)
- ⚠️ Keras/TensorFlow (port required)

**Pro Edition includes:**
- ✅ HuggingFace Trainer integration
- ✅ PyTorch Lightning support (coming soon)

---

## Configuration

### How do I choose patience_steps?

**Rule of thumb:**

| Dataset Difficulty | patience_steps |
|-------------------|----------------|
| Easy (MNIST-like) | 3 |
| Medium (CIFAR-like) | 4 |
| Hard (ImageNet-like) | 5 |
| Fine-tuning (BERT) | 2 |

**Start with 3**, then:
- Increase if stops too early
- Decrease if runs too long

---

### How do I choose min_delta?

**Rule of thumb:**

| Use Case | min_delta |
|----------|-----------|
| Fast convergence | 0.01 (1% improvement) |
| Moderate | 0.005 (0.5% improvement) |
| Noisy validation | 0.005 or lower |

**Start with 0.01**, then:
- Lower if stops too early (too strict)
- Raise if stops too late (too lenient)

---

### What if my validation loss is noisy?

**Increase EMA smoothing:**

```python
rca = ResonantCallback(
    patience_steps=3,
    min_delta=0.005,   # More lenient
    ema_alpha=0.5,     # More smoothing (default: 0.3)
)
```

**Or use larger validation set** to reduce noise.

---

### Should I use the same config for all datasets?

**No!** Different datasets need different configs:

**Easy datasets** (MNIST, Fashion-MNIST):
- Short patience (3)
- Strict improvement (0.01)

**Hard datasets** (ImageNet, large NLP):
- Long patience (5)
- Lenient improvement (0.005)

**Fine-tuning** (BERT):
- Very short patience (2)
- Lenient improvement (0.005)

See [Configuration Guide](./API.md#configuration-examples) for details.

---

## Usage

### How do I add RCA to my existing code?

**Just 3 lines:**

```python
from resonant_callback import ResonantCallback

# 1. Initialize
rca = ResonantCallback(checkpoint_dir='./checkpoints', patience_steps=3, min_delta=0.01)

# 2. Call after validation
for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)
    rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
    
    # 3. Check stop
    if rca.should_stop():
        break
```

---

### Can I use RCA with learning rate schedulers?

**Yes!** RCA's adaptive LR works alongside most schedulers:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

for epoch in range(max_epochs):
    # Train & validate...
    
    # RCA (will reduce LR when needed)
    rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
    
    # Your scheduler
    scheduler.step()
    
    if rca.should_stop():
        break
```

**RCA's LR reduction happens in addition to your scheduler.**

---

### How do I disable checkpoint saving?

```python
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    save_checkpoints=False  # Disable checkpointing
)
```

**Use case:** Hyperparameter search where you don't need checkpoints.

---

### Can I manually load the best checkpoint later?

**Yes:**

```python
# After training
if rca.load_best_checkpoint(model):
    print("Best model loaded!")
else:
    print("Checkpoint not found, using current model")
```

Or manually:
```python
import torch

checkpoint = torch.load('./checkpoints/best_model_epoch9_loss0.027813.pt')
model.load_state_dict(checkpoint)
```

---

## Troubleshooting

### Training never stops

**Symptoms:**
- Runs to max epochs
- β never reaches >0.70

**Causes & Solutions:**

1. **patience_steps too high**
   ```python
   patience_steps=3  # instead of 5
   ```

2. **min_delta too strict**
   ```python
   min_delta=0.005  # instead of 0.01
   ```

3. **Model needs more epochs**
   ```python
   max_epochs=100  # instead of 50
   ```

---

### Stops too early

**Symptoms:**
- Stops after few epochs
- Loss still decreasing

**Causes & Solutions:**

1. **min_delta too lenient**
   ```python
   min_delta=0.01  # instead of 0.005
   ```

2. **patience_steps too low**
   ```python
   patience_steps=4  # instead of 3
   ```

3. **Validation loss too noisy**
   ```python
   ema_alpha=0.5  # more smoothing
   ```

---

### Checkpoint not found

**Symptoms:**
- Warning: "Best checkpoint not found"
- Using final model instead

**Solutions:**

1. **Check directory exists:**
   ```python
   import os
   os.makedirs('./checkpoints', exist_ok=True)
   ```

2. **Verify disk space:**
   ```bash
   df -h  # Check free space
   ```

3. **Check permissions:**
   ```bash
   ls -la ./checkpoints
   ```

---

### Out of memory (OOM)

**Symptoms:**
- CUDA OOM error
- Training crashes

**Solutions:**

1. **Reduce batch size:**
   ```python
   batch_size=32  # instead of 64
   ```

2. **Use gradient accumulation:**
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(train_loader):
       loss = loss / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Train on CPU:**
   ```python
   device = torch.device('cpu')
   ```

---

### Different results each run

**Cause:** Random seed not set

**Solution:**
```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

---

## Understanding RCA

### What does β > 0.70 mean?

**β (beta) measures loss stability:**

- **β = 0.0-0.5:** Large oscillations, still learning
- **β = 0.5-0.7:** Moderate oscillations, converging
- **β > 0.70:** Stable plateau, likely converged ✅

**When β crosses 0.70**, RCA considers it a plateau and may stop if no improvement occurs.

---

### Why does RCA reduce learning rate?

**Adaptive LR helps escape plateaus:**

When validation loss plateaus:
1. RCA reduces learning rate (e.g., 0.001 → 0.0005)
2. Smaller steps may reveal further improvements
3. If no improvement → reduce again
4. After max reductions → stop training

**This gives the model a chance** to improve before stopping.

---

### What is EMA and why use it?

**EMA = Exponential Moving Average**

Smooths noisy validation loss:
```python
loss_smooth = alpha * loss_new + (1 - alpha) * loss_old
```

**Benefits:**
- Reduces impact of outliers
- Makes β/ω metrics more stable
- Better stopping decisions

**Default:** `ema_alpha=0.3` works well for most cases.

---

### Can RCA detect overfitting?

**Yes, indirectly!**

When validation loss plateaus while training loss keeps decreasing:
- β increases (plateau detected)
- RCA stops training
- Best checkpoint is loaded (from before overfitting)

**Example:** CIFAR-10 achieved +1.35% better accuracy because RCA stopped before overfitting.

---

## Advanced Topics

### Can I use custom stopping criteria?

**Yes!** Combine RCA with your own logic:

```python
rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
stats = rca.get_statistics()

# Stop if RCA says stop OR custom condition
if rca.should_stop() or (stats['beta'] > 0.90 and val_acc > 0.95):
    print("Stopping: RCA or custom condition met")
    break
```

---

### Can I use RCA for multi-task learning?

**Partially.** RCA currently tracks single metric (validation loss).

**Workaround:**
```python
# Combine multiple losses
combined_loss = loss_task1 + loss_task2 + loss_task3
rca(val_loss=combined_loss, model=model, optimizer=optimizer, epoch=epoch)
```

**Pro Edition:** Supports multi-metric stopping (coming soon).

---

### Can I use RCA with DistributedDataParallel?

**Yes!** RCA works with DDP:

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[rank])

# RCA on rank 0 only
if rank == 0:
    rca = ResonantCallback(checkpoint_dir='./checkpoints')
    
for epoch in range(max_epochs):
    # Train & validate...
    
    if rank == 0:
        rca(val_loss=val_loss, model=model.module, optimizer=optimizer, epoch=epoch)
        should_stop = rca.should_stop()
    else:
        should_stop = False
    
    # Broadcast stop signal
    should_stop = torch.tensor(should_stop).to(device)
    dist.broadcast(should_stop, src=0)
    
    if should_stop:
        break
```

---

### How does RCA compare to Optuna's pruning?

**Different purposes:**

**Optuna pruning:**
- Stops *bad* hyperparameter configs early
- Used during hyperparameter optimization
- Compares multiple trials

**RCA:**
- Stops *converged* training early
- Used during single training run
- Saves compute on any training

**Use both:**
```python
# Optuna trial
for epoch in range(max_epochs):
    # Train & validate...
    
    # RCA
    rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
    
    # Optuna pruning
    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    # RCA early stop
    if rca.should_stop():
        break
```

---

## Performance

### Does RCA add overhead?

**Negligible!** RCA analysis takes <1ms per epoch.

**Breakdown:**
- β/ω calculation: ~0.5ms
- Checkpoint saving: ~10-50ms (only when improving)
- Total overhead: <0.1% of training time

---

### Can I use RCA for real-time training?

**Yes!** RCA overhead is minimal.

**However:** Early stopping is most useful for long training runs (>10 epochs). For very short training (<5 epochs), RCA may not trigger.

---

### What's the best batch size for RCA?

**RCA is batch-size agnostic.** Use whatever works for your model.

**However:**
- Larger batches → more stable validation loss → more reliable β/ω
- Smaller batches → noisier validation loss → may need higher `ema_alpha`

---

## Community & Support

### How do I report bugs?

**GitHub Issues:** https://github.com/.../issues

**Include:**
1. RCA version (`pip show resonant-learner`)
2. PyTorch version (`torch.__version__`)
3. Minimal reproducible example
4. Error message / unexpected behavior

---

### How do I request features?

**GitHub Discussions:** https://github.com/.../discussions

**Or:** Open a feature request issue

**Popular requests:**
- Multi-metric stopping → Pro Edition has this
- TensorBoard integration → Coming soon
- PyTorch Lightning support → Coming soon

---

### How can I contribute?

**We welcome contributions!**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

**See:** [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines

---

### Where can I find more examples?

**In the repository:**
- `examples/mnist_rca.py` - MNIST baseline
- `examples/fashion_mnist_rca.py` - Fashion-MNIST
- `examples/cifar10_rca.py` - CIFAR-10
- `examples/hf_bert_glue.py` - BERT fine-tuning

**Want more?** Open an issue requesting specific examples!

---

## Still Have Questions?

**Check these resources:**

1. **Quick Start:** [QUICKSTART.md](./QUICKSTART.md)
2. **API Docs:** [API.md](./API.md)
3. **Scientific Report:** [SCIENTIFIC_VALIDATION_REPORT.md](./SCIENTIFIC_VALIDATION_REPORT.md)
4. **GitHub Issues:** https://github.com/.../issues
5. **GitHub Discussions:** https://github.com/.../discussions

**For Professional Edition:**
- Email: zakelj.damjan@gmail.com
- See: [Edition Comparison](./EDITIONS_COMPARISON.md)

---

**Can't find your question?** Open an issue and we'll add it to the FAQ! 💙

---

*"Stop training when your model converges, not epochs later."* 🌊✨
