#!/usr/bin/env python3
"""
CIFAR-10 Example with Resonant Learner - Community Edition

Demonstrates intelligent early stopping on CIFAR-10 natural image classification.

Usage:
    # Baseline (no RCA)
    python cifar10_rca.py --baseline --epochs 60
    
    # With RCA
    python cifar10_rca.py --epochs 60
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


class CifarNet(nn.Module):
    """CNN for CIFAR-10 natural image classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_epoch(model, device, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, device, val_loader, criterion):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 with Resonant Learner - Community Edition')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Maximum number of epochs (default: 60)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate (default: 0.001)')
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline without RCA')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_cifar10',
                       help='Directory for checkpoints')
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Print header
    print("\n" + "=" * 70)
    if args.baseline:
        print("🔵 BASELINE MODE - CIFAR-10 Natural Image Classification 🔵")
    else:
        print("🌊 RCA MODE - CIFAR-10 Natural Image Classification 🌊")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial learning rate: {args.lr}")
    print(f"Max epochs: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Data loading
    print("\n📂 Loading CIFAR-10 dataset...")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"✅ Dataset loaded!")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Test samples: {len(test_dataset):,}")
    print(f"   Number of classes: {len(class_names)}")
    print(f"   Classes: {', '.join(class_names)}")
    print(f"   Image size: 32×32×3 (color)")
    print(f"   Augmentation: Random crop + horizontal flip")
    
    # Model
    print("\n🏗️  Building model...")
    model = CifarNet(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created!")
    print(f"   Architecture: Custom CNN (3 conv layers + 2 FC layers)")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # RCA Setup
    rca = None
    if not args.baseline:
        print("\n🌊 Initializing Resonant Convergence Analysis (RCA)...")
        rca = ResonantCallback(
            checkpoint_dir=args.checkpoint_dir,
            patience_steps=4,  # More patience for harder dataset
            min_delta=0.005,   # Smaller threshold for harder dataset
            ema_alpha=0.3,
            max_lr_reductions=2,
            lr_reduction_factor=0.5,
            min_lr=1e-6,
            verbose=True
        )
        print("✅ RCA initialized!")
        print(f"   Checkpoint dir: {args.checkpoint_dir}")
        print(f"   Patience: 4 epochs (harder dataset)")
        print(f"   Min improvement: 0.5%")
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
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_acc = validate(model, device, test_loader, criterion)
        
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
                    best_val_loss, best_val_acc = validate(model, device, test_loader, criterion)
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
