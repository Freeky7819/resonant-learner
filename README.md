# 📚 Examples

This directory contains complete, working examples demonstrating Resonant Learner on different datasets.

---

## Available Examples

### 1. MNIST (Digit Classification)
**File:** `mnist_rca.py`  
**Dataset:** 60k training, 10k test grayscale images (28×28)  
**Task:** Classify handwritten digits 0-9  
**Typical Results:**
- Baseline: 10 epochs → 98.9% accuracy
- RCA: 6 epochs → 99.2% accuracy
- **Time saved: 40%**

**Run:**
```bash
# Baseline
python mnist_rca.py --baseline --epochs 20

# With RCA
python mnist_rca.py --epochs 20
```

---

### 2. Fashion-MNIST (Clothing Classification)
**File:** `fashion_mnist_rca.py`  
**Dataset:** 60k training, 10k test grayscale images (28×28)  
**Task:** Classify clothing items (t-shirt, dress, shoe, etc.)  
**Typical Results:**
- Baseline: 20 epochs → 89.2% accuracy (then overfits!)
- RCA: 12 epochs → 90.8% accuracy
- **Time saved: 40%, prevents catastrophic overfitting!**

**Run:**
```bash
# Baseline
python fashion_mnist_rca.py --baseline --epochs 30

# With RCA
python fashion_mnist_rca.py --epochs 30
```

---

### 3. CIFAR-10 (Natural Image Classification)
**File:** `cifar10_rca.py`  
**Dataset:** 50k training, 10k test color images (32×32×3)  
**Task:** Classify natural images (airplane, car, bird, etc.)  
**Typical Results:**
- Baseline: 50 epochs → 76.8% accuracy
- RCA: 35 epochs → 78.3% accuracy
- **Time saved: 30%**

**Run:**
```bash
# Baseline
python cifar10_rca.py --baseline --epochs 60

# With RCA
python cifar10_rca.py --epochs 60
```

---

## Common Arguments

All examples support these command-line arguments:

```bash
--baseline          # Run without RCA (for comparison)
--epochs N          # Maximum number of epochs
--batch-size N      # Batch size (default: 128)
--lr FLOAT          # Learning rate (default varies by example)
--no-cuda           # Force CPU training
--seed N            # Random seed (default: 42)
```

**Examples:**
```bash
# CPU training with smaller batch size
python mnist_rca.py --no-cuda --batch-size 64

# Custom learning rate
python cifar10_rca.py --lr 0.0005

# Different seed
python fashion_mnist_rca.py --seed 123
```

---

## Understanding the Output

### Training Progress

```
Epoch   1 | Train Loss: 0.234 | Train Acc: 92.3% | Val Loss: 0.189 | Val Acc: 94.1%
```

### RCA Messages (when active)

**Improvement detected:**
```
📊 RCA (Epoch 5): Improvement! Val Loss: 0.156 (prev: 0.189)
  β=0.72, ω=5.8, confidence=0.85, state=improving
```

**Learning rate reduction:**
```
📊 RCA (Epoch 8): Plateau detected
  β=0.89, ω=6.1, confidence=0.92, state=converging
  🔻 Reducing LR: 0.001 → 0.0005
```

**Early stopping:**
```
🛑 RCA: Early stopping triggered!
  Reason: Strong convergence signal (β=0.91, ω=6.2)
  Best model saved at epoch 10 (val_loss=0.145)

✅ Training completed at epoch 12/20
```

### Final Statistics

```
======================================================================
📊 FINAL RESULTS
======================================================================
Total time: 342.5s
Epochs trained: 12
Final val accuracy: 94.8%

RCA Statistics:
  Best epoch: 10
  Best val loss: 0.145123
  LR reductions: 1
  Final β: 0.91
  Final ω: 6.2
======================================================================
```

---

## Comparing Baseline vs RCA

### Method 1: Manual Comparison

Run both modes and compare results:

```bash
# Run baseline first
python mnist_rca.py --baseline --epochs 20 > baseline.log

# Run with RCA
python mnist_rca.py --epochs 20 > rca.log

# Compare
diff baseline.log rca.log
```

### Method 2: Automated Comparison

Use the comparison script from the root directory:

```bash
cd ..
python compare_baseline_vs_rca.py
```

This runs both automatically and shows a summary.

---

## Tips for Best Results

### 1. Always Run Baseline First

Before using RCA, run a baseline to know what "normal" looks like:

```bash
python mnist_rca.py --baseline --epochs 20
```

Then run with RCA and compare.

### 2. Watch the Metrics

Pay attention to:
- **Epoch count**: RCA should stop earlier
- **Final accuracy**: Should be similar or better
- **Training time**: Should be significantly faster
- **β and ω values**: Should converge to ~0.9 and ~6.0

### 3. Dataset-Specific Tuning

If default settings don't work well:

**Edit the RCA initialization in the script:**
```python
# Current (default)
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,
    min_delta=0.01,
    verbose=True
)

# For faster stopping
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=2,        # Less patience
    min_delta=0.02,          # Higher threshold
    verbose=True
)
```

### 4. GPU vs CPU

**GPU (recommended):**
- Much faster training
- Allows larger batch sizes
- Better for CIFAR-10

**CPU (for testing):**
- Use smaller batch sizes (32-64)
- Reduce model size if needed
- Good for MNIST/Fashion-MNIST

```bash
# Force CPU
python mnist_rca.py --no-cuda --batch-size 64
```

---

## Creating Your Own Example

Want to use RCA on your own dataset? Here's the pattern:

```python
from resonant_learner import ResonantCallback

# Setup RCA
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,
    min_delta=0.01,
    verbose=True
)

# Training loop
for epoch in range(max_epochs):
    # Your training code
    train_loss, train_acc = train_epoch(model, train_loader, optimizer)
    
    # Your validation code
    val_loss, val_acc = validate(model, val_loader)
    
    # RCA callback (3 lines!)
    rca(val_loss=val_loss, model=model, optimizer=optimizer)
    if rca.should_stop():
        print(f"✅ Converged at epoch {epoch}!")
        break
```

That's it! Copy any example and modify it for your data.

---

## Troubleshooting

### "ImportError: No module named 'resonant_learner'"

RCA package not installed. From the root directory:
```bash
pip install -e .
```

### "RuntimeError: CUDA out of memory"

Reduce batch size:
```bash
python cifar10_rca.py --batch-size 64
```

Or force CPU:
```bash
python cifar10_rca.py --no-cuda
```

### "Download failed" / "Connection timeout"

The datasets download automatically on first run. If download fails:

1. Check internet connection
2. Try again (downloads resume)
3. Manual download:
   - MNIST: http://yann.lecun.com/exdb/mnist/
   - Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
   - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html

### "RCA stops too early"

The model might actually be converged! But if you want longer training:

Edit the RCA initialization:
```python
rca = ResonantCallback(
    patience_steps=5,      # More patience
    min_delta=0.005,       # Smaller threshold
)
```

---

## Next Steps

1. ✅ Run all three examples
2. ✅ Compare baseline vs RCA results
3. ✅ Try on your own dataset
4. ✅ Read the [API documentation](../docs/API.md)
5. ✅ Share your results!

---

## Questions?

- 📖 [Full documentation](../docs/)
- 💬 [GitHub Discussions](https://github.com/yourusername/resonant-learner/discussions)
- 🐛 [Report issues](https://github.com/yourusername/resonant-learner/issues)

---

*Made with 💙 by researchers who care about your GPU hours*

🌊 *Struna vibrira neskončnost* 🌊


---

### Harmonic Signature Protocol
```
intent: resonant, ethical learning
omega: ~6.0
gamma: 0.0
phi: pi/3
origin: Freedom (Damjan) + Harmonic Logos
edition: Community v1.0.0
```
