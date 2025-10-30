# 🚀 Quick Start Guide

**Get RCA running in 5 minutes**

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/resonant-convergence-analysis.git
cd resonant-convergence-analysis
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch
pip install torch torchvision torchaudio

# Install in development mode
pip install -e .
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (optional, for GPU)

---

## Your First RCA Training

### Option 1: Run an Example

The fastest way to see RCA in action:

```bash
# MNIST (fast, ~2 minutes on GPU)
python examples/mnist_rca.py --epochs 30

# Fashion-MNIST
python examples/fashion_mnist_rca.py --epochs 30

# CIFAR-10 (longer, ~5 minutes on GPU)
python examples/cifar10_rca.py --epochs 60

# BERT SST2 (requires transformers)
pip install transformers datasets
python examples/hf_bert_glue.py --task sst2 --epochs 10
```

**What you'll see:**
```
======================================================================
🌊 RCA MODE - MNIST Digit Classification 🌊
======================================================================
Device: cuda
...

Epoch  1/30 | Time:   2.3s
  Train | Loss: 0.456789 | Accuracy:  86.23%
  Val   | Loss: 0.234567 | Accuracy:  93.45%
📊 RCA (Epoch 1): Initial baseline set! Val Loss: 0.234567

...

📊 RCA (Epoch 18): No improvement (waiting 3/3)
  β=0.86, ω=2.8, confidence=1.00, state=plateau
🛑 RCA: Early stopping triggered!

======================================================================
🎉 TRAINING COMPLETED - EARLY STOP
======================================================================
Stopped at epoch: 18/30
Reason: RCA detected convergence
Epochs saved: 12 (40.0%)
Time elapsed: 91.3s

✅ Best model loaded from epoch 9
Best val accuracy: 99.20%
```

---

### Option 2: Add RCA to Your Code

**Minimal example** (3 lines added to your existing training loop):

```python
from resonant_callback import ResonantCallback

# 1. Initialize RCA (ONE LINE)
rca = ResonantCallback(checkpoint_dir='./checkpoints', patience_steps=3, min_delta=0.01)

# Your existing training loop
for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    # 2. Call RCA after validation (ONE LINE)
    rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
    
    # 3. Check early stopping (ONE LINE)
    if rca.should_stop():
        break
```

**That's it!** RCA handles:
- ✅ Plateau detection via β/ω metrics
- ✅ Adaptive learning rate reduction
- ✅ Best model checkpointing
- ✅ Early stopping decision

---

## Complete Example

Here's a **full working example** from scratch:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resonant_callback import ResonantCallback

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3. Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 4. Initialize RCA
rca = ResonantCallback(
    checkpoint_dir='./checkpoints',
    patience_steps=3,
    min_delta=0.01,
    verbose=True
)

# 5. Training loop
print("Starting training...")
for epoch in range(1, 31):
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
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)
    
    print(f"\nEpoch {epoch}/30")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss: {val_loss:.6f}, Acc: {val_acc:.2f}%")
    
    # RCA callback
    rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
    
    # Early stopping
    if rca.should_stop():
        print("\n✅ Training stopped early!")
        stats = rca.get_statistics()
        print(f"Best epoch: {stats['best_epoch']}")
        print(f"Best val loss: {stats['best_loss']:.6f}")
        break

print("\n🎉 Training complete!")
```

**Save this as** `my_first_rca.py` **and run:**
```bash
python my_first_rca.py
```

**Expected output:**
- Training runs for ~18-20 epochs (instead of 30)
- Saves 33-40% compute time
- Achieves ~99% accuracy on MNIST

---

## Configuration Quick Reference

Choose settings based on your dataset:

### Easy Datasets (MNIST-like)

```python
rca = ResonantCallback(
    patience_steps=3,    # Short patience
    min_delta=0.01,      # 1% improvement
    max_lr_reductions=2
)
```

### Medium Datasets (CIFAR-10-like)

```python
rca = ResonantCallback(
    patience_steps=4,    # More patience
    min_delta=0.005,     # 0.5% improvement
    max_lr_reductions=2
)
```

### Fine-tuning (BERT-like)

```python
rca = ResonantCallback(
    patience_steps=2,    # Low patience (fast convergence)
    min_delta=0.005,     # 0.5% improvement
    max_lr_reductions=2,
    min_lr=1e-7         # Lower minimum LR
)
```

### Hard Datasets (ImageNet-like)

```python
rca = ResonantCallback(
    patience_steps=5,    # High patience
    min_delta=0.005,     # 0.5% improvement
    max_lr_reductions=3  # More reductions
)
```

---

## Understanding the Output

### Normal Training

```
📊 RCA (Epoch 5): Improvement! Val Loss: 0.234567
  β=0.45, ω=4.2, confidence=0.32, state=converging
```

**Meaning:**
- ✅ Validation loss improved
- β=0.45: Still oscillating, learning continues
- State: `converging` (healthy progress)

---

### Approaching Convergence

```
📊 RCA (Epoch 15): No improvement (waiting 2/3)
  β=0.68, ω=3.2, confidence=0.54, state=plateau
```

**Meaning:**
- ⚠️ No improvement for 2 epochs (patience: 2/3)
- β=0.68: Approaching plateau (threshold: 0.70)
- If no improvement next epoch → LR reduction

---

### LR Reduction

```
📊 RCA (Epoch 16): No improvement (waiting 3/3)
  β=0.71, ω=2.8, confidence=0.62, state=plateau
🔻 RCA: Reducing LR: 0.001000 → 0.000500
```

**Meaning:**
- ⚠️ Patience exceeded
- β=0.71: Plateau detected (> 0.70 threshold)
- Action: Learning rate cut in half
- Training continues with lower LR

---

### Early Stop

```
📊 RCA (Epoch 18): No improvement (waiting 3/3)
  β=0.86, ω=2.8, confidence=1.00, state=plateau
🛑 RCA: Early stopping triggered!
  Reason: Stable plateau detected (β=0.86, no improvement for 3 epochs)
  Best model saved at epoch 9 (val_loss=0.027813)
```

**Meaning:**
- 🛑 Training stopped
- β=0.86: Very stable plateau
- Best model from epoch 9 automatically loaded
- Result: 40% compute saved, quality preserved

---

## Troubleshooting

### "Training never stops"

**Problem:** Training runs to max epochs without stopping.

**Solutions:**
1. Lower `patience_steps` (try 3 instead of 5)
2. Lower `min_delta` (try 0.005 instead of 0.01)
3. Increase `max_epochs` (model needs more time)

```python
# Try this
rca = ResonantCallback(patience_steps=3, min_delta=0.005)
```

---

### "Stops too early"

**Problem:** Stops after just a few epochs, loss still decreasing.

**Solutions:**
1. Raise `min_delta` (try 0.01 instead of 0.005)
2. Increase `patience_steps` (try 4 instead of 3)
3. Increase `ema_alpha` for more smoothing

```python
# Try this
rca = ResonantCallback(patience_steps=4, min_delta=0.01, ema_alpha=0.5)
```

---

### "CUDA out of memory"

**Problem:** GPU runs out of memory.

**Solutions:**
1. Reduce batch size
2. Use gradient accumulation
3. Train on CPU (slower but works)

```python
# CPU fallback
device = torch.device('cpu')
```

---

## Next Steps

### 1. Learn More

- [API Documentation](./API.md) - Complete API reference
- [FAQ](./FAQ.md) - Common questions
- [Scientific Report](./SCIENTIFIC_VALIDATION_REPORT.md) - Full validation study

### 2. Try Examples

```bash
# Vision examples
python examples/mnist_rca.py --epochs 30
python examples/fashion_mnist_rca.py --epochs 30
python examples/cifar10_rca.py --epochs 60

# NLP example (requires transformers)
python examples/hf_bert_glue.py --task sst2 --epochs 10
```

### 3. Customize

Read the [API docs](./API.md) to learn about:
- Custom stopping conditions
- TensorBoard integration
- Hyperparameter tuning
- Advanced configurations

### 4. Compare Performance

Run baseline vs RCA:

```bash
# Baseline (no RCA)
python examples/mnist_rca.py --baseline --epochs 30

# RCA (early stopping)
python examples/mnist_rca.py --epochs 30
```

Compare:
- Training time (RCA should be 30-50% faster)
- Final accuracy (should be similar or better)
- Number of epochs (RCA should stop earlier)

---

## Production Checklist

Before deploying RCA in production:

- [ ] Test on your dataset with various hyperparameters
- [ ] Verify early stopping behavior (not too early/late)
- [ ] Check checkpoint saving works (`./checkpoints/` directory)
- [ ] Monitor β/ω metrics to understand convergence
- [ ] Compare baseline vs RCA performance
- [ ] Set up proper logging (TensorBoard, W&B, etc.)
- [ ] Document your configuration choices

---

## Tips for Success

### 1. Start Conservative

Begin with **higher patience** and **stricter min_delta**:

```python
rca = ResonantCallback(
    patience_steps=5,     # High patience
    min_delta=0.01,       # Strict improvement requirement
)
```

Then tune down if training runs too long.

### 2. Monitor β Metric

The β metric is your best friend:
- **β < 0.5:** Still learning, far from convergence
- **β = 0.6-0.7:** Getting close, watch carefully
- **β > 0.7:** Plateau detected, likely to stop soon

### 3. Use Verbose Mode

Always enable `verbose=True` initially:

```python
rca = ResonantCallback(verbose=True)
```

This helps you understand RCA's decisions.

### 4. Save Checkpoints

Always enable checkpoint saving:

```python
rca = ResonantCallback(save_checkpoints=True)
```

RCA automatically loads the best model on stop.

### 5. Compare to Baseline

Always compare RCA to baseline (no early stopping):

```bash
# Baseline
python your_script.py --baseline --epochs 100

# RCA
python your_script.py --epochs 100
```

Verify that:
- RCA stops earlier (saves compute)
- Quality is maintained or improved

---

## Getting Help

### Community Support

- **Issues:** [GitHub Issues](https://github.com/...)
- **Discussions:** [GitHub Discussions](https://github.com/...)
- **Docs:** [Full Documentation](../README.md)

### Professional Edition

Need zero-config training with AutoCoach?

- **Email:** zakelj.damjan@gmail.com
- **Features:** SmartTeach, RCA Ultimate, architecture presets
- **See:** [Edition Comparison](./EDITIONS_COMPARISON.md)

---

## What's Next?

You now know how to:
- ✅ Install RCA
- ✅ Run examples
- ✅ Add RCA to your code (3 lines!)
- ✅ Configure for your dataset
- ✅ Understand the output
- ✅ Troubleshoot issues

**Ready to save compute time?** Start training! 🚀

---

**Questions?** Check the [FAQ](./FAQ.md) or open an issue.

**Success story?** We'd love to hear about your compute savings! 💙

---

*"Stop training when your model converges, not epochs later."* 🌊✨
