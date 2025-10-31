#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal ImageNet RCA Trainer
Compatible with all ResonantCallback versions (Phase 4 → Phase 5.2)
Author: Freedom (Damjan) / Harmonic Logos
"""

import os
import time
import sys
from pathlib import Path

# Add parent directory to path for resonant_callback import
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# ---------------------------------------------------------------------
# Try to import ResonantCallback safely
# ---------------------------------------------------------------------
try:
    from resonant_learner import ResonantCallback
except ImportError:
    ResonantCallback = None
    print("⚠️  Warning: ResonantCallback not found — running in BASELINE mode.")

# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./data", type=str)
parser.add_argument("--tiny", action="store_true")
parser.add_argument("--arch", default="resnet18", type=str)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--batch-size", default=256, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--workers", default=4, type=int)
args = parser.parse_args()

# ---------------------------------------------------------------------
# Device info
# ---------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🚀 Using GPU: {gpu}\n   Memory: {mem:.1f}GB")
else:
    print("⚠️ Using CPU fallback.")

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
print(f"\n📁 Loading data from: {args.data}")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64 if args.tiny else 224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
val_transform = transforms.Compose([
    transforms.Resize(72 if args.tiny else 256),
    transforms.CenterCrop(64 if args.tiny else 224),
    transforms.ToTensor(),
    normalize,
])

train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.data, "val"), transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

print("✅ Dataset loaded!")
print(f"   Training samples: {len(train_dataset):,}")
print(f"   Validation samples: {len(val_dataset):,}")
print(f"   Number of classes: {len(train_dataset.classes)}")

# ---------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------
print(f"\n🏗️  Building model: {args.arch}")
model = getattr(torchvision.models, args.arch)(weights=None, num_classes=len(train_dataset.classes))
model = model.to(device)
if device == "cuda":
    cudnn.benchmark = True

print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
writer = SummaryWriter("runs")

# ---------------------------------------------------------------------
# RCA initialization (auto-detect version)
# ---------------------------------------------------------------------
rca = None
if ResonantCallback is not None:
    try:
        # Newer versions
        rca = ResonantCallback(patience_steps=5, min_delta=0.005,
                               reduce_factor=0.5, max_lr_reductions=3)
    except TypeError:
        # Older versions (no reduce_factor)
        rca = ResonantCallback(patience_steps=5, min_delta=0.005,
                               max_lr_reductions=3)

    print("\n🌊 RCA enabled!")
    print("   Patience: 5 epochs")
    print("   Min delta: 0.0050")
else:
    print("\n⚪ RCA disabled (baseline mode).")

# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate(model, val_loader):
    model.eval()
    correct1 = correct5 = total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, pred = outputs.topk(5, 1, True, True)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct1 += correct[:, :1].sum().item()
            correct5 += correct[:, :5].sum().item()
            total += targets.size(0)
    val_loss /= total
    top1 = 100. * correct1 / total
    top5 = 100. * correct5 / total
    return val_loss, top1, top5

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
print("""
======================================================================
🌊 RCA MODE 🌊
======================================================================
Architecture: {}
Dataset: Tiny-ImageNet
Device: {}
Batch size: {}
Initial LR: {}
Max epochs: {}
Workers: {}
======================================================================
""".format(args.arch, device, args.batch_size, args.lr, args.epochs, args.workers))

print("🚀 Starting training...\n")
best_val_loss = float("inf")

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss = 0.0
    correct1 = correct5 = total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        _, pred = outputs.topk(5, 1, True, True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        correct1 += correct[:, :1].sum().item()
        correct5 += correct[:, :5].sum().item()
        total += targets.size(0)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f} | Top-1: {100.*correct1/total:.2f}% | Top-5: {100.*correct5/total:.2f}%")

    val_loss, val_top1, val_top5 = validate(model, val_loader)
    train_loss = epoch_loss / total
    train_top1 = 100. * correct1 / total
    train_top5 = 100. * correct5 / total
    elapsed = time.time() - start_time

    print("\n======================================================================")
    print(f"Epoch {epoch:3d}/{args.epochs} Summary (Time: {elapsed:.1f}s)")
    print("======================================================================")
    print(f"Train | Loss: {train_loss:.4f} | Top-1: {train_top1:6.2f}% | Top-5: {train_top5:6.2f}%")
    print(f"Val   | Loss: {val_loss:.4f} | Top-1: {val_top1:6.2f}% | Top-5: {val_top5:6.2f}%")
    print("======================================================================\n")

    # Save checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_model_epoch{epoch}_loss{val_loss:.6f}.pt")
        print(f"💾 Checkpoint saved: best_model_epoch{epoch}_loss{val_loss:.6f}.pt")
        if rca is not None:
            print(f"📊 RCA (Epoch {epoch}): Improvement! Val Loss: {val_loss:.6f}")

    # RCA logic
    if rca is not None:
        rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
        stop_flag = False
        if hasattr(rca, "should_stop") and callable(rca.should_stop):
            stop_flag = rca.should_stop()
        elif hasattr(rca, "stop_training"):
            stop_flag = bool(rca.stop_training)
        elif hasattr(rca, "state") and getattr(rca, "state", "") == "stop":
            stop_flag = True
        if stop_flag:
            print(f"\n🛑 RCA early stop triggered at epoch {epoch}/{args.epochs}")
            
            # Load best model checkpoint if available
            stats = rca.get_statistics()
            best_checkpoint = f"best_model_epoch{stats['best_epoch']}_loss{stats['best_loss']:.6f}.pt"
            if os.path.exists(best_checkpoint):
                model.load_state_dict(torch.load(best_checkpoint))
                print(f"✅ Best model loaded from epoch {stats['best_epoch']}")
                # Re-evaluate with best model
                final_val_loss, final_top1, final_top5 = validate(model, val_loader)
                print(f"Best val accuracy: Top-1={final_top1:.2f}%, Top-5={final_top5:.2f}%")
            else:
                print(f"⚠️  Best checkpoint not found, using final model")
            
            break

    scheduler.step()

print("\n✅ Training finished.")
writer.close()
