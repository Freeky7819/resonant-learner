#!/usr/bin/env python3
"""
MNIST Example with Resonant Learner - Community Edition

Demonstrates intelligent early stopping on MNIST digit classification.

Usage:
    # Baseline (no RCA)
    python mnist_rca.py --baseline --epochs 20
    
    # With RCA
    python mnist_rca.py --epochs 20
"""

import argparse
import time
from pathlib import Path
import sys

# Add parent directory to path for resonant_callback import
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resonant_learner import ResonantCallback


class Net(nn.Module):
    """Simple CNN for MNIST digit classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_epoch(model, device, train_loader, optimizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, device, val_loader):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum')
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='MNIST with Resonant Learner - Community Edition')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                       help='Test batch size (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Maximum number of epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline without RCA')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_mnist',
                       help='Directory for checkpoints')
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Print header
    print("\n" + "=" * 70)
    if args.baseline:
        print("🔵 BASELINE MODE - MNIST Digit Classification 🔵")
    else:
        print("🌊 RCA MODE - MNIST Digit Classification 🌊")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"Batch size: {args.batch_size}")
    print(f"Test batch size: {args.test_batch_size}")
    print(f"Initial learning rate: {args.lr}")
    print(f"Max epochs: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Data loading
    print("\n📂 Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    
    print(f"✅ Dataset loaded!")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    print(f"   Number of classes: {len(train_dataset.classes)}")
    print(f"   Image size: 28×28 (grayscale)")
    
    # Model
    print("\n🏗️  Building model...")
    model = Net().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # RCA Setup
    rca = None
    if not args.baseline:
        print("\n🌊 Initializing Resonant Convergence Analysis (RCA)...")
        rca = ResonantCallback(
            checkpoint_dir=args.checkpoint_dir,
            patience_steps=3,
            min_delta=0.01,
            ema_alpha=0.3,
            max_lr_reductions=2,
            lr_reduction_factor=0.5,
            min_lr=1e-6,
            verbose=True
        )
        print("✅ RCA initialized!")
        print(f"   Checkpoint dir: {args.checkpoint_dir}")
        print(f"   Patience: 3 epochs")
        print(f"   Min improvement: 1.0%")
        print(f"   EMA alpha: 0.3")
        print(f"   Max LR reductions: 2")
    
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70 + "\n")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer)
        
        # Validate
        val_loss, val_acc = validate(model, device, test_loader)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"Epoch {epoch:3d}/{args.epochs} | Time: {epoch_time:5.1f}s")
        print(f"  Train | Loss: {train_loss:.6f} | Accuracy: {train_acc:6.2f}%")
        print(f"  Val   | Loss: {val_loss:.6f} | Accuracy: {val_acc:6.2f}%")
        
        # RCA callback
        if rca is not None:
            rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
            
            if rca.should_stop():
                total_time = time.time() - start_time
                stats = rca.get_statistics()
                
                # Load best model checkpoint
                best_checkpoint_path = Path(args.checkpoint_dir) / f"best_model_epoch{stats['best_epoch']}_loss{stats['best_loss']:.6f}.pt"
                if best_checkpoint_path.exists():
                    model.load_state_dict(torch.load(best_checkpoint_path))
                    # Re-evaluate with best model
                    best_val_loss, best_val_acc = validate(model, device, test_loader)
                    print("\n" + "=" * 70)
                    print("🎉 TRAINING COMPLETED - EARLY STOP")
                    print("=" * 70)
                    print(f"Stopped at epoch: {epoch}/{args.epochs}")
                    print(f"Reason: RCA detected convergence")
                    saved = args.epochs - epoch
                    print(f"Epochs saved: {saved} ({saved/args.epochs*100:.1f}%)")
                    print(f"Time elapsed: {total_time:.1f}s")
                    print(f"\n✅ Best model loaded from epoch {stats['best_epoch']}")
                    print(f"Best val accuracy: {best_val_acc:.2f}%")
                else:
                    print("\n" + "=" * 70)
                    print("🎉 TRAINING COMPLETED - EARLY STOP")
                    print("=" * 70)
                    print(f"Stopped at epoch: {epoch}/{args.epochs}")
                    print(f"Reason: RCA detected convergence")
                    saved = args.epochs - epoch
                    print(f"Epochs saved: {saved} ({saved/args.epochs*100:.1f}%)")
                    print(f"Time elapsed: {total_time:.1f}s")
                    print(f"⚠️  Best checkpoint not found, using final model")
                    print(f"Final val accuracy: {val_acc:.2f}%")
                
                print("\n📊 RCA Statistics:")
                print(f"  Best epoch: {stats['best_epoch']}")
                print(f"  Best val loss: {stats['best_loss']:.6f}")
                print(f"  LR reductions: {stats['lr_reductions']}")
                print(f"  Final β: {stats['beta']:.3f}")
                print(f"  Final ω: {stats['omega']:.2f}")
                print(f"  State: {stats['state']}")
                print("=" * 70)
                break
        
        print()  # Empty line between epochs
    else:
        # Training completed normally (all epochs)
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED - ALL EPOCHS")
        print("=" * 70)
        print(f"Total epochs: {args.epochs}")
        print(f"Time elapsed: {total_time:.1f}s")
        print(f"Final val accuracy: {val_acc:.2f}%")
        
        if rca is not None:
            stats = rca.get_statistics()
            print("\n📊 RCA Statistics:")
            print(f"  Best epoch: {stats['best_epoch']}")
            print(f"  Best val loss: {stats['best_loss']:.6f}")
            print(f"  LR reductions: {stats['lr_reductions']}")
            print(f"  Final β: {stats['beta']:.3f}")
            print(f"  Final ω: {stats['omega']:.2f}")
            print(f"  State: {stats['state']}")
        
        print("=" * 70)


if __name__ == '__main__':
    main()
