#!/usr/bin/env python3
"""
CIFAR-10/100 Example with Resonant Learner + SmartTeach (PRODUCTION READY!)

Supports both CIFAR-10 (10 classes) and CIFAR-100 (100 classes)
with RCA, SmartTeach, and baseline modes.

🔥 FIXED: RCA now ACTS (LR reduction + early stop), not just logs!
🔥 FIXED: CIFAR-100 head mismatch (100 classes)
🔥 FIXED: Aligned parameters with successful tests
🔥 FIXED: No more import spam from DataLoader workers!

Usage:
    # CIFAR-100 Baseline
    python examples/cifar_example.py --dataset cifar100 --epochs 50 --batch-size 256 --seed 42 --use-rca 0
    
    # CIFAR-100 with RCA (ACTIONABLE!)
    python examples/cifar_example.py --dataset cifar100 --epochs 50 --batch-size 256 --seed 42 --use-rca 1 --ema-alpha 0.7 --patience-steps 2
    
    # CIFAR-100 Full Stack (RCA + SmartTeach)
    python examples/cifar_example.py --dataset cifar100 --epochs 50 --batch-size 256 --seed 42 --use-rca 1 --use-smart-teach 1

Author: Freedom & Harmonic Logos
Fixed by: Claude Analytikos φ₂
Date: October 27, 2025
"""

# CRITICAL: Add parent directory to path BEFORE any other imports
import sys
import os
import random
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent if _script_dir.name == 'examples' else _script_dir
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Additional path guards for resonant_learner module
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("resonant_learner"))

import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Import ResonantCallback SILENTLY (no print spam from workers!)
ResonantCallback = None
HAS_RCA = False

try:
    from resonant_learner import ResonantCallback
    HAS_RCA = True
    # ✅ NO PRINT HERE - moved to main() to avoid worker spam!
except ImportError:
    pass  # Will be handled in main()
except Exception:
    pass  # Will be handled in main()

# ═══════════════════════════════════════════════════════════════════
# 🤖 AutoCoach Integration
# ═══════════════════════════════════════════════════════════════════
try:
    from resonant_learner.auto_learner import AutoCoach
    HAS_AUTOCOACH = True
except ImportError:
    AutoCoach = None
    HAS_AUTOCOACH = False

# SmartTeacher will be imported in main() to avoid DataLoader worker spam


# ----------------------------
# SmartTeacher
# ----------------------------
class CifarNet(nn.Module):
    """
    CNN for CIFAR-10/100 (32x32 RGB images).
    
    Args:
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)  # ✅ Adaptive output!

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        x = self.dropout1(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten and FC
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Return RAW LOGITS (not log_softmax) for compatibility with cross_entropy
        return x


# ----------------------------
# Training & Evaluation
# ----------------------------
def train_epoch(model, device, loader, optimizer, coach=None, teacher=None, epoch=0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # Works with raw logits
        
        # 💾 SAVE ORIGINAL LOSS FOR LOGGING (before modification)
        original_loss = loss.item()
        
        # ═══════════════════════════════════════════════════════════════════
        # 🤖 HOOK 2: AutoCoach Smart Feedback (in training loop)
        # ═══════════════════════════════════════════════════════════════════
        if coach is not None:
            t_phase = batch_idx / max(1, len(loader))
            feedback = coach.smart_feedback(loss.item(), t_phase)
            loss = loss * (1.0 + feedback)
        elif teacher is not None:
            # Manual SmartTeach (backward compat)
            t_phase = epoch * len(loader) + batch_idx
            feedback = teacher.step(loss.item(), t_phase)
            loss = loss + feedback
        
        loss.backward()
        optimizer.step()
        
        # 🔥 USE ORIGINAL LOSS FOR STATISTICS (not modified loss!)
        loss_sum += original_loss * data.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += data.size(0)
    
    return loss_sum / total, 100.0 * correct / total


def eval_epoch(model, device, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction="sum")
            loss_sum += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += data.size(0)
    
    return loss_sum / total, 100.0 * correct / total


# ----------------------------
# Main
# ----------------------------
def main():
    # ═══════════════════════════════════════════════════════════════════
    # SmartTeacher Import (in main to avoid DataLoader worker spam!)
    # ═══════════════════════════════════════════════════════════════════
    global SmartTeacher, HAS_SMARTTEACH
    SmartTeacher = None
    HAS_SMARTTEACH = False
    try:
        from resonant_learner.smart_teach import SmartTeacher
        HAS_SMARTTEACH = True
    except ImportError:
        pass  # Silent fail - it's optional (PRO feature)
    
    parser = argparse.ArgumentParser(description="CIFAR-10/100 + Resonant Learner + SmartTeach (PRODUCTION!)")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["cifar10", "cifar100"],
                        help="Choose dataset: cifar10 (10 classes) or cifar100 (100 classes)")
    
    # Model selection
    parser.add_argument("--model", type=str, default="cifarnet",
                        choices=["cifarnet", "resnet18"],
                        help="Choose model: cifarnet (custom CNN) or resnet18 (from torchvision)")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # 🔥 NEW: RCA action parameters (aligned with successful tests)
    parser.add_argument("--eval-every-epochs", type=int, default=2,
                        help="Evaluate and check RCA every N epochs")
    parser.add_argument("--ema-alpha", type=float, default=0.7,
                        help="RCA EMA smoothing parameter")
    parser.add_argument("--patience-steps", type=int, default=10,
                        help="RCA patience: epochs without improvement before LR reduction")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="RCA minimum RELATIVE improvement threshold (0.005 = 0.5% improvement required)")
    parser.add_argument("--max-lr-reductions", type=int, default=2,
                        help="Maximum LR reductions before early stop (default: 2)")
    parser.add_argument("--amp", type=int, default=1,
                        help="Use automatic mixed precision (1=on, 0=off)")
    parser.add_argument("--num-workers", type=int, default=6,
                        help="DataLoader workers (0 for Windows compatibility, 6 for speed)")
    
    # Mode selection
    parser.add_argument("--baseline", action="store_true", 
                        help="Disable RCA and SmartTeach")
    parser.add_argument("--use-rca", type=int, default=1, 
                        help="1=RCA on, 0=off")
    parser.add_argument("--use-smart-teach", type=int, default=0, 
                        help="1=SmartTeach on, 0=off")
    
    # AutoCoach arguments
    parser.add_argument("--no-autocoach", action="store_true",
                        help="Disable AutoCoach (use manual RCA/SmartTeach settings)")
    parser.add_argument("--preset", type=str, default="auto",
                        help="AutoCoach preset: auto, bert, cnn, gpt, vit, default")
    
    # Paths
    parser.add_argument("--logdir", type=str, default="checkpoints_cifar")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory to store datasets")
    
    # SmartTeach parameters
    parser.add_argument("--rc-alpha", type=float, default=0.05)
    parser.add_argument("--rc-omega", type=float, default=6.0)
    parser.add_argument("--rc-phi", type=float, default=0.3)
    parser.add_argument("--teach-ema-alpha", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # ✅ Print import status ONLY in main process (no worker spam!)
    if HAS_RCA:
        print("✅ ResonantCallback imported successfully!")
    else:
        print("⚠️  ResonantCallback not available")
        print("⚠️  Install with: pip install -e .")
        print("⚠️  Running in baseline mode only")
        if args.use_rca == 1:
            print("⚠️  Forcing --use-rca 0 (RCA not available)")
            args.use_rca = 0
    
    # 🔥 Deterministic training for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # 🔥 GPU/CPU detection message (like ultimate version)
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    elif not args.no_cuda and device.type == "cpu":
        print("⚠️  CUDA not available; training on CPU")
    else:
        print("💻 Training on CPU (--no-cuda specified)")
    
    # Dataset with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # ✅ Dataset selection with proper class count
    dataset_name = args.dataset.lower()
    
    if dataset_name == "cifar100":
        print("📚 Loading CIFAR-100 dataset (100 classes)...")
        num_classes = 100
        train_ds = datasets.CIFAR100(args.data_dir, train=True,  download=True, transform=transform_train)
        test_ds  = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=transform_test)
    else:  # cifar10
        print("📦 Loading CIFAR-10 dataset (10 classes)...")
        num_classes = 10
        train_ds = datasets.CIFAR10(args.data_dir, train=True,  download=True, transform=transform_train)
        test_ds  = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform_test)
    
    # 🔥 OPTIMIZED DataLoaders - MUCH FASTER!
    # pin_memory=True → speeds up CPU→GPU transfer
    # persistent_workers=True → keeps workers alive (faster on Windows)
    # prefetch_factor=2 → preload next batches
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,  
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),  # Only if using GPU
        persistent_workers=(args.num_workers > 0),  # Keep workers alive
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    test_loader = DataLoader(
        test_ds,  
        batch_size=args.test_batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    print(f"✅ Dataset loaded: {len(train_ds)} train, {len(test_ds)} test samples")
    
    # ✅ Model selection with correct number of classes
    print(f"🏗️ Building model: {args.model} (num_classes={num_classes})")
    
    if args.model == "cifarnet":
        model = CifarNet(num_classes=num_classes).to(device)
        print(f"[Model] CifarNet created with {num_classes} output classes")
    elif args.model == "resnet18":
        # Use torchvision ResNet18
        from torchvision.models import resnet18
        model = resnet18(weights=None)  # No pretrained weights (random init)
        
        # Adapt final layer for num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        
        print(f"[Model] ResNet18 created with {num_classes} output classes")
        print(f"[Model] Note: ResNet18 works with 32×32 input (CIFAR native size)")
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # 🔥 DEBUG: Verify model is on GPU
    if device.type == 'cuda':
        print(f"[Debug] Model device: {next(model.parameters()).device}")
        print(f"[Debug] GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # ═══════════════════════════════════════════════════════════════════
    # 🤖 HOOK 1: AutoCoach Initialization (after model/optimizer)
    # ═══════════════════════════════════════════════════════════════════
    coach = None
    # 🔥 FIXED: Only use AutoCoach if NOT explicitly requesting RCA/SmartTeach
    explicit_rca = args.use_rca == 1
    explicit_smart = args.use_smart_teach == 1
    use_autocoach = not (args.baseline or explicit_rca or explicit_smart or 
                         (hasattr(args, 'no_autocoach') and args.no_autocoach))
    
    if use_autocoach and HAS_AUTOCOACH:
        try:
            coach = AutoCoach.from_args(args)
            print(f"🤖 AutoCoach: Preset '{coach.preset_name}' loaded!")
        except Exception as e:
            print(f"⚠️  AutoCoach failed: {e}")
            use_autocoach = False
    
    # Fallback to manual RCA/SmartTeach if AutoCoach not available/disabled
    if not use_autocoach:
        # 🔥 RCA callback with ACTION HOOKS (not just logging!)
        rca = None
        use_rca = (args.use_rca == 1) and (not args.baseline) and HAS_RCA
        
        if use_rca:
            try:
                # ✅ Pass ALL RCA parameters!
                rca = ResonantCallback(
                    checkpoint_dir=args.logdir,
                    enable_tensorboard=False,
                    verbose=True,
                    # 🔥 ACTION PARAMETERS - CRITICAL!
                    ema_alpha=args.ema_alpha,
                    patience_steps=args.patience_steps,
                    min_delta=args.min_delta,
                    max_lr_reductions=args.max_lr_reductions
                )
                print("✅ ResonantCallback initialized (ACTIONABLE MODE)")
                print(f"   → EMA α={args.ema_alpha}, patience={args.patience_steps}, min_δ={args.min_delta}, max_LR_reductions={args.max_lr_reductions}")
            except Exception as e:
                print(f"⚠️  RCA initialization failed: {e}")
                print("⚠️  Continuing without RCA actions")
                rca = None
                use_rca = False
    else:
        # AutoCoach handles everything
        rca = None
        use_rca = False
        use_smart_teach = False  # 👈 ADDED: Initialize when AutoCoach is active
    
    # SmartTeacher (if AutoCoach not used)
    teacher = None
    if not use_autocoach:
        use_smart_teach = (args.use_smart_teach == 1) and (not args.baseline) and HAS_SMARTTEACH
        if use_smart_teach and SmartTeacher is not None:
            teacher = SmartTeacher(
                alpha=args.rc_alpha, 
                omega=args.rc_omega, 
                phi=args.rc_phi, 
                ema_alpha=args.teach_ema_alpha
            )
        elif args.use_smart_teach == 1:
            print("⚠️  SmartTeach requested but not available - continuing without it")
            use_smart_teach = False
    
    print("=" * 70)
    mode_str = []
    if coach:
        mode_str.append(f"AutoCoach({coach.preset_name})")
    else:
        if use_rca:
            mode_str.append("RCA (ACTIONABLE!)")
        if teacher:
            mode_str.append("SmartTeach")
        if not mode_str:
            mode_str.append("BASELINE")
    mode_display = " + ".join(mode_str)
    
    if coach:
        print(f"🤖 AUTOCOACH MODE: {mode_display} 🤖")
    elif use_rca and teacher:
        print(f"🌊🔥 FULL STACK: {mode_display} 🔥🌊")
    elif use_rca:
        print(f"🌊 RCA MODE (WITH ACTIONS!): {mode_display} 🌊")
    elif teacher:
        print(f"🔥 SMARTTEACH MODE: {mode_display} 🔥")
    else:
        print(f"🔵 BASELINE MODE: {mode_display} 🔵")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()} ({num_classes} classes)")
    print(f"Model: {args.model.upper()}")
    print(f"Device: {device.type}")
    print(f"Batch size: {args.batch_size}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Initial LR: {args.lr}")
    print(f"Max epochs: {args.epochs}")
    if use_rca:
        print(f"RCA params: EMA_α={args.ema_alpha}, patience={args.patience_steps}, min_δ={args.min_delta}, max_LR_reductions={args.max_lr_reductions}")
        print(f"Eval interval: every {args.eval_every_epochs} epochs")
    if use_smart_teach:
        print(f"SmartTeach: α={args.rc_alpha}, ω={args.rc_omega}, φ={args.rc_phi}")
    print(f"Checkpoint dir: {args.logdir}")
    print(f"Seed: {args.seed} (deterministic)")
    print("=" * 70)
    
    start_time = time.time()
    best_test_loss = float("inf")
    best_test_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, coach, teacher, epoch)
        test_loss, test_acc = eval_epoch(model, device, test_loader)
        epoch_time = time.time() - epoch_start
        
        # 🔥 GPU stats for monitoring
        gpu_mem = ""
        if device.type == 'cuda':
            gpu_mem = f" | GPU: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB"
        
        print(
            f"\nEpoch {epoch:3d} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%{gpu_mem}"
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # 🤖 HOOK 3: AutoCoach After Eval (validation + early stopping)
        # ═══════════════════════════════════════════════════════════════════
        if coach is not None and (epoch % max(1, args.eval_every_epochs) == 0):
            coach.after_eval(test_loss, train_loss)
            coach.maybe_adjust(optimizer, model)
            
            if coach.should_stop:
                print("🛑 AutoCoach: Early stopping triggered")
                time_elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print("🎉 TRAINING COMPLETED (AutoCoach Early Stop)")
                print(f"{'='*70}")
                print(f"Stopped at epoch: {epoch}/{args.epochs}")
                saved = args.epochs - epoch
                print(f"Epochs saved: {saved} ({saved/args.epochs*100:.1f}%)")
                print(f"Time elapsed: {time_elapsed:.1f}s ({time_elapsed/60:.1f} min)")
                print(f"Final test accuracy: {test_acc:.2f}%")
                print(f"{'='*70}")
                break
        
        # 🔥 SIMPLE BOOLEAN STOP CHECK (alternative interface) - Manual RCA
        if rca and (epoch % max(1, args.eval_every_epochs) == 0):
            if rca(test_loss, model, optimizer, epoch):  # Fixed order: val_loss, model, optimizer, epoch
                time_elapsed = time.time() - start_time
                print(f"\n{'='*70}")
                print("🚨 RCA STOP SIGNAL RECEIVED — ENDING TRAINING")
                print(f"{'='*70}")
                print(f"Stopped at epoch: {epoch}/{args.epochs}")
                saved = args.epochs - epoch
                print(f"Epochs saved: {saved} ({saved/args.epochs*100:.1f}%)")
                print(f"Time elapsed: {time_elapsed:.1f}s ({time_elapsed/60:.1f} min)")
                print(f"Final test accuracy: {test_acc:.2f}%")
                
                if rca:
                    try:
                        stats = rca.get_statistics()
                        print("\n📊 RCA Statistics:")
                        print(f"  Best loss: {stats.get('best_loss', 0.0):.6f}")
                        print(f"  LR reductions: {stats.get('lr_reductions', 0)}")
                        if 'loss_reduction' in stats:
                            print(f"  Loss reduction: {stats.get('loss_reduction', 0.0):.1f}%")
                    except Exception:
                        pass
                print(f"{'='*70}")
                break
        
        # Best model tracking (always save best regardless of mode)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_acc = test_acc
            
            # Save checkpoint
            checkpoint_path = Path(args.logdir) / f"best_epoch{epoch}_loss{best_test_loss:.6f}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': best_test_loss,
                'test_acc': best_test_acc,
            }, checkpoint_path)
    
    # Final summary
    time_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ TRAINING FINISHED")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()} ({num_classes} classes)")
    print(f"Mode: {mode_display}")
    print(f"Total epochs: {epoch}")
    print(f"Time elapsed: {time_elapsed:.1f}s ({time_elapsed/60:.1f} min)")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
