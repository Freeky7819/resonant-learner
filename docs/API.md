# 🌊 RCA API Reference - Community Edition

**Complete API documentation for ResonantCallback**

---

## Overview

The `ResonantCallback` class is the core component of RCA Community Edition. It provides intelligent early stopping based on resonance analysis of validation loss.

**Key Features:**
- 🎯 Automatic plateau detection via β/ω metrics
- 🔧 Adaptive learning rate reduction
- 💾 Best model checkpointing
- 📊 Training statistics tracking
- ⚡ Production-validated performance

---

## Quick Example

```python
from resonant_callback import ResonantCallback

# Initialize
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,
    min_delta=0.01,
    verbose=True
)

# Training loop
for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # RCA callback
    rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
    
    if rca.should_stop():
        print("Early stopping triggered!")
        break
```

---

## Class: `ResonantCallback`

### Constructor

```python
ResonantCallback(
    checkpoint_dir: str = 'checkpoints',
    patience: int = 10,
    patience_steps: Optional[int] = None,
    min_delta: float = 0.0,
    ema_alpha: float = 0.3,
    max_lr_reductions: int = 2,
    lr_reduction_factor: float = 0.5,
    min_lr: float = 1e-6,
    verbose: bool = True,
    save_checkpoints: bool = True,
)
```

### Parameters

#### Core Parameters

**`checkpoint_dir`** : `str`, default=`'checkpoints'`
- Directory to save model checkpoints
- Created automatically if doesn't exist
- Best model saved as: `best_model_epoch{N}_loss{X}.pt`

**`patience_steps`** : `int`, default=`None`
- Number of epochs to wait without improvement before LR reduction
- If `None`, uses legacy `patience` parameter
- Recommended: 3 (easy), 4 (medium), 5 (hard datasets)

**`min_delta`** : `float`, default=`0.0`
- Minimum improvement in validation loss to reset patience
- Example: `0.01` means 1% improvement required
- Recommended: `0.01` (easy), `0.005` (medium/hard datasets)

#### Adaptive Learning Rate

**`max_lr_reductions`** : `int`, default=`2`
- Maximum number of learning rate reductions before stopping
- After this many reductions, training stops
- Typical: 2-3 reductions

**`lr_reduction_factor`** : `float`, default=`0.5`
- Factor to multiply learning rate by when reducing
- Example: `0.5` means cut LR in half
- Must be between 0 and 1

**`min_lr`** : `float`, default=`1e-6`
- Minimum learning rate threshold
- Training stops if LR falls below this value
- Prevents infinitesimally small learning rates

#### Analysis Parameters

**`ema_alpha`** : `float`, default=`0.3`
- Exponential moving average smoothing factor
- Higher = more smoothing (less sensitive to noise)
- Range: 0.1-0.5 typical

#### Other Parameters

**`verbose`** : `bool`, default=`True`
- Print RCA analysis after each epoch
- Shows β, ω, state, and decisions
- Disable for silent operation

**`save_checkpoints`** : `bool`, default=`True`
- Enable/disable checkpoint saving
- Useful to disable for hyperparameter search

**`patience`** : `int`, default=`10` (legacy)
- Legacy parameter, prefer `patience_steps`
- Used if `patience_steps=None`

---

## Methods

### `__call__(val_loss, model, optimizer, epoch)`

Main callback method called after each epoch.

**Parameters:**
- `val_loss` : `float` or `torch.Tensor`
  - Validation loss for current epoch
  - Automatically converted to float

- `model` : `torch.nn.Module`
  - PyTorch model to checkpoint
  - Only used if `save_checkpoints=True`

- `optimizer` : `torch.optim.Optimizer`
  - PyTorch optimizer
  - Used for learning rate reduction

- `epoch` : `int`
  - Current epoch number (1-indexed)

**Returns:** `None`

**Example:**
```python
val_loss = validate(model, val_loader)
rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
```

---

### `should_stop()`

Check if training should stop.

**Returns:** `bool`
- `True` if early stopping triggered
- `False` otherwise

**Example:**
```python
if rca.should_stop():
    print("Training completed!")
    break
```

---

### `get_statistics()`

Get current training statistics.

**Returns:** `dict` with keys:
- `'best_epoch'` : Epoch with lowest validation loss
- `'best_loss'` : Best validation loss achieved
- `'lr_reductions'` : Number of LR reductions performed
- `'beta'` : Current β (resonance amplitude)
- `'omega'` : Current ω (resonance frequency)
- `'state'` : Current training state (`'plateau'`, `'converging'`, etc.)

**Example:**
```python
stats = rca.get_statistics()
print(f"Best epoch: {stats['best_epoch']}")
print(f"Best loss: {stats['best_loss']:.6f}")
print(f"Final β: {stats['beta']:.3f}")
```

---

### `load_best_checkpoint(model)`

Load the best model checkpoint.

**Parameters:**
- `model` : `torch.nn.Module`
  - Model to load weights into

**Returns:** `bool`
- `True` if checkpoint loaded successfully
- `False` if checkpoint not found

**Example:**
```python
if rca.load_best_checkpoint(model):
    print("Best model loaded!")
else:
    print("Using final model weights")
```

---

## Resonance Metrics

### Beta (β) - Resonance Amplitude

**Range:** `[0, 1]`

**Interpretation:**
- `β → 1.0`: Very stable, minimal oscillations (converged)
- `β > 0.70`: Plateau detected (stop condition)
- `β < 0.50`: Large oscillations (still learning)

**Formula:**
```python
β = 1 - (max_oscillation / avg_loss)
```

### Omega (ω) - Resonance Frequency

**Range:** `[0, 12]`

**Interpretation:**
- `ω ≈ 6.0`: Optimal oscillation frequency
- `ω → 0`: Very slow changes
- `ω > 10`: High-frequency oscillations (instability)

**Formula:**
```python
ω = 2π × (zero_crossings / window_size)
```

### Training States

**`'initializing'`**
- Not enough data yet (< 10 epochs)
- No stopping decisions made

**`'converging'`**
- Loss decreasing, β increasing
- Healthy training progress

**`'plateau'`**
- β > 0.70, minimal improvement
- Potential early stop candidate

**`'oscillating'`**
- β < 0.70, significant oscillations
- Still learning

**`'diverging'`**
- Loss increasing, training unstable
- Rare, indicates problems

---

## Configuration Examples

### Easy Datasets (MNIST, Fashion-MNIST)

```python
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,        # Short patience
    min_delta=0.01,          # 1% improvement required
    ema_alpha=0.3,
    max_lr_reductions=2,
    lr_reduction_factor=0.5,
    verbose=True
)
```

**Expected behavior:**
- Stops around 40-50% of max epochs
- 2 LR reductions typical
- β reaches 0.85+ at stop

---

### Medium Datasets (CIFAR-10, CIFAR-100)

```python
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=4,        # More patience
    min_delta=0.005,         # 0.5% improvement
    ema_alpha=0.3,
    max_lr_reductions=2,
    lr_reduction_factor=0.5,
    verbose=True
)
```

**Expected behavior:**
- Stops around 60-80% of max epochs
- 2 LR reductions typical
- β reaches 0.75-0.85 at stop

---

### Fine-tuning (BERT, Pre-trained Models)

```python
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=2,        # Low patience (fast convergence)
    min_delta=0.005,         # 0.5% improvement
    ema_alpha=0.3,
    max_lr_reductions=2,
    lr_reduction_factor=0.5,
    min_lr=1e-7,            # Lower min LR for fine-tuning
    verbose=True
)
```

**Expected behavior:**
- Stops around 70% of max epochs
- 2 LR reductions typical
- β may be lower (0.70-0.75) due to fast convergence

---

### Hard Datasets (ImageNet, Large NLP)

```python
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=5,        # High patience
    min_delta=0.005,         # 0.5% improvement
    ema_alpha=0.3,
    max_lr_reductions=3,     # More reductions allowed
    lr_reduction_factor=0.5,
    verbose=True
)
```

**Expected behavior:**
- Stops around 70-90% of max epochs
- 3 LR reductions typical
- β reaches 0.70-0.80 at stop

---

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from resonant_callback import ResonantCallback

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize RCA
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,
    min_delta=0.01,
    max_lr_reductions=2,
    verbose=True
)

# Training loop
max_epochs = 100
for epoch in range(1, max_epochs + 1):
    # Train
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    # Print progress
    print(f"Epoch {epoch}/{max_epochs}")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss: {val_loss:.6f}")
    
    # RCA callback
    rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
    
    # Check early stopping
    if rca.should_stop():
        print("\n🛑 Early stopping triggered!")
        stats = rca.get_statistics()
        print(f"Best epoch: {stats['best_epoch']}")
        print(f"Best val loss: {stats['best_loss']:.6f}")
        print(f"LR reductions: {stats['lr_reductions']}")
        print(f"Final β: {stats['beta']:.3f}")
        print(f"Final ω: {stats['omega']:.2f}")
        
        # Load best model
        if rca.load_best_checkpoint(model):
            print("✅ Best model loaded!")
        
        break

print("\n✅ Training completed!")
```

---

## Verbose Output Example

When `verbose=True`, RCA prints analysis after each epoch:

```
Epoch 15/30
  Train Loss: 0.234567
  Val Loss: 0.345678

📊 RCA (Epoch 15): No improvement (waiting 2/3)
  β=0.68, ω=3.2, confidence=0.54, state=converging

Epoch 16/30
  Train Loss: 0.223456
  Val Loss: 0.345012

📊 RCA (Epoch 16): No improvement (waiting 3/3)
  β=0.71, ω=2.8, confidence=0.62, state=plateau
🔻 RCA: Reducing LR: 0.001000 → 0.000500

Epoch 17/30
  Train Loss: 0.219876
  Val Loss: 0.344567

📊 RCA (Epoch 17): Improvement! Val Loss: 0.344567
  β=0.72, ω=2.6, confidence=0.65, state=plateau

Epoch 18/30
  Train Loss: 0.218234
  Val Loss: 0.344890

📊 RCA (Epoch 18): No improvement (waiting 1/3)
  β=0.73, ω=2.5, confidence=0.68, state=plateau

Epoch 19/30
  Train Loss: 0.217123
  Val Loss: 0.345234

📊 RCA (Epoch 19): No improvement (waiting 2/3)
  β=0.74, ω=2.4, confidence=0.71, state=plateau

Epoch 20/30
  Train Loss: 0.216456
  Val Loss: 0.345567

📊 RCA (Epoch 20): No improvement (waiting 3/3)
  β=0.75, ω=2.3, confidence=0.73, state=plateau
🔻 RCA: Reducing LR: 0.000500 → 0.000250
🛑 RCA: Early stopping triggered!
  Reason: Stable plateau detected (β=0.75, no improvement for 3 epochs)
  Best model saved at epoch 17 (val_loss=0.344567)
```

---

## Troubleshooting

### Training Never Stops

**Symptoms:**
- Training runs to max epochs
- β never reaches > 0.70

**Possible causes:**
1. `patience_steps` too high
2. `min_delta` too strict
3. Dataset too hard (needs more epochs)

**Solutions:**
```python
# Try lower patience
patience_steps=3  # instead of 5

# Or more lenient improvement threshold
min_delta=0.005  # instead of 0.01

# Or increase max epochs
max_epochs=100  # instead of 50
```

---

### Stops Too Early

**Symptoms:**
- Stops after just a few epochs
- Validation loss still decreasing

**Possible causes:**
1. `min_delta` too lenient
2. `patience_steps` too low
3. Noisy validation loss

**Solutions:**
```python
# Require more improvement
min_delta=0.01  # instead of 0.005

# Increase patience
patience_steps=4  # instead of 3

# Smooth validation loss more
ema_alpha=0.5  # instead of 0.3
```

---

### Checkpoint Not Found

**Symptoms:**
- Warning: "Best checkpoint not found"
- Using final model instead of best

**Possible causes:**
1. `checkpoint_dir` doesn't exist or not writable
2. Disk full
3. Checkpoint file deleted

**Solutions:**
```python
# Check directory exists and is writable
import os
os.makedirs('./checkpoints', exist_ok=True)

# Or disable checkpointing (not recommended)
save_checkpoints=False
```

---

## Performance Tips

### For Hyperparameter Search

```python
# Disable verbose output and checkpointing for speed
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,
    min_delta=0.01,
    verbose=False,           # Silent operation
    save_checkpoints=False   # Skip disk I/O
)
```

### For Production Training

```python
# Enable all features
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,
    min_delta=0.01,
    verbose=True,            # Monitor progress
    save_checkpoints=True    # Save best model
)
```

---

## Integration with Other Tools

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# In training loop
rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
stats = rca.get_statistics()

writer.add_scalar('RCA/beta', stats['beta'], epoch)
writer.add_scalar('RCA/omega', stats['omega'], epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
```

### Weights & Biases

```python
import wandb

wandb.init(project="my-project")

# In training loop
rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
stats = rca.get_statistics()

wandb.log({
    'val_loss': val_loss,
    'rca_beta': stats['beta'],
    'rca_omega': stats['omega'],
    'rca_state': stats['state']
}, step=epoch)
```

---

## Advanced Usage

### Custom Early Stopping Condition

```python
# Stop if β > threshold OR custom condition
rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
stats = rca.get_statistics()

if rca.should_stop() or (stats['beta'] > 0.90 and val_acc > 0.95):
    print("Custom stop condition met!")
    break
```

### Manual LR Control

```python
# RCA will still reduce LR, but you can override
rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)

# Apply custom LR schedule on top
if epoch % 10 == 0:
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.9
```

---

## API Changelog

### v5.0 (Current)
- ✅ Fixed β threshold (0.70 instead of 0.75)
- ✅ Production-validated on 4 datasets
- ✅ Improved checkpoint naming
- ✅ Better verbose output

### v4.0
- ❌ Bug: Missed β=0.70-0.75 plateaus
- Introduced `patience_steps` parameter
- Added EMA smoothing

### v3.0
- Initial public release
- Basic RCA functionality

---

## License

MIT License - see [LICENSE](../LICENSE) for details

---

## Support

- **Documentation:** This file + [README](../README.md)
- **Examples:** See [examples/](../examples/) directory
- **Issues:** [GitHub Issues](https://github.com/...)
- **Discussions:** [GitHub Discussions](https://github.com/...)

---

**Status:** ✅ Production Ready  
**Version:** v5.0  
**Validation:** NVIDIA L40S GPU + PyTorch 2.9.0

*"Intelligent early stopping via resonance analysis."* 🌊✨
