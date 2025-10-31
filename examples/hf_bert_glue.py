#!/usr/bin/env python3
"""
BERT GLUE Example with Resonant Learner - Community Edition

Tests RCA on NLP fine-tuning task (sentiment classification).
Uses manual PyTorch loop for full RCA control.

Expected results:
- Automatic LR adjustment
- Early stopping on validation plateau
- Competitive accuracy with fewer epochs

Usage:
    # Baseline (no RCA)
    python hf_bert_glue.py --task sst2 --epochs 5 --baseline
    
    # RCA (with intelligent stopping)
    python hf_bert_glue.py --task sst2 --epochs 5
    
    # DistilBERT (less VRAM)
    python hf_bert_glue.py --model distilbert-base-uncased --epochs 5
"""

import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path for resonant_callback import
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Resonant Learner
from resonant_learner import ResonantCallback

# HuggingFace imports
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    from datasets import load_dataset
except ImportError:
    print("❌ transformers/datasets not installed:")
    print("   pip install transformers datasets")
    sys.exit(1)


def get_glue_loaders(task, model_name, batch_size, test_batch_size):
    """Load GLUE task and create train/val loaders."""
    # Load dataset
    dataset = load_dataset("glue", task)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize function
    def tokenize_function(examples):
        if task == "sst2":
            return tokenizer(examples["sentence"], padding="max_length", 
                           truncation=True, max_length=128)
        elif task == "mrpc":
            return tokenizer(examples["sentence1"], examples["sentence2"],
                           padding="max_length", truncation=True, max_length=128)
        else:
            raise NotImplementedError(f"Task {task} not implemented")
    
    # Tokenize
    tokenized = dataset.map(tokenize_function, batched=True)
    
    # Remove unused columns
    tokenized = tokenized.remove_columns(
        ["sentence"] if task == "sst2" else ["sentence1", "sentence2", "idx"]
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    
    # Create loaders
    train_loader = DataLoader(
        tokenized["train"], 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        tokenized["validation"], 
        batch_size=test_batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, tokenizer


def train_epoch(model, device, train_loader, optimizer, scheduler, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, device, loader, criterion):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='BERT GLUE with Resonant Learner - Community Edition')
    
    # Model & Task
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       choices=['bert-base-uncased', 'distilbert-base-uncased'],
                       help='Pretrained model to use')
    parser.add_argument('--task', type=str, default='sst2',
                       choices=['sst2', 'mrpc'],
                       help='GLUE task to fine-tune on')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5,
                       help='Maximum number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                       help='Test batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Linear warmup steps (default: 500)')
    
    # RCA
    parser.add_argument('--baseline', action='store_true',
                       help='Run baseline without RCA')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_bert',
                       help='Directory for checkpoints')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Print header
    print("\n" + "=" * 70)
    if args.baseline:
        print(f"🔵 BASELINE MODE - {args.model} on {args.task.upper()} 🔵")
    else:
        print(f"🌊 RCA MODE - {args.model} on {args.task.upper()} 🌊")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model: {args.model}")
    print(f"Task: {args.task.upper()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Max epochs: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Data loading
    print(f"\n📂 Loading GLUE {args.task.upper()}...")
    train_loader, val_loader, tokenizer = get_glue_loaders(
        args.task, args.model, args.batch_size, args.test_batch_size
    )
    
    print(f"✅ Dataset loaded!")
    print(f"   Training samples: {len(train_loader.dataset):,}")
    print(f"   Validation samples: {len(val_loader.dataset):,}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    if args.task == "sst2":
        print(f"   Task: Sentiment classification (positive/negative)")
    elif args.task == "mrpc":
        print(f"   Task: Paraphrase detection")
    
    # Model
    print(f"\n🏗️  Loading {args.model}...")
    num_labels = 2  # Binary classification for both sst2 and mrpc
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model loaded!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.1f}MB (fp32)")
    
    # Optimizer & Scheduler
    print(f"\n⚙️  Setting up optimizer and scheduler...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps, 
        num_training_steps=total_steps
    )
    
    print(f"✅ Optimizer configured!")
    print(f"   Total training steps: {total_steps:,}")
    print(f"   Warmup steps: {args.warmup_steps}")
    print(f"   Effective warmup: {100 * args.warmup_steps / total_steps:.1f}%")
    
    # RCA Setup
    rca = None
    if not args.baseline:
        print("\n🌊 Initializing Resonant Convergence Analysis (RCA)...")
        rca = ResonantCallback(
            checkpoint_dir=args.checkpoint_dir,
            patience_steps=2,  # Fewer patience for BERT (shorter training)
            min_delta=0.005,
            ema_alpha=0.3,
            max_lr_reductions=2,
            lr_reduction_factor=0.5,
            min_lr=1e-7,
            verbose=True
        )
        print("✅ RCA initialized!")
        print(f"   Checkpoint dir: {args.checkpoint_dir}")
        print(f"   Patience: 2 epochs")
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
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, scheduler, criterion
        )
        
        # Validate
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"Epoch {epoch:2d}/{args.epochs} | Time: {epoch_time:5.1f}s")
        print(f"  Train | Loss: {train_loss:.6f} | Accuracy: {train_acc:6.2f}%")
        print(f"  Val   | Loss: {val_loss:.6f} | Accuracy: {val_acc:6.2f}%")
        
        # RCA callback
        if rca is not None:
            # Note: For BERT we evaluate every epoch, not every N steps
            # RCA analyzes the validation loss to detect convergence
            rca(val_loss=val_loss, model=model, optimizer=optimizer, epoch=epoch)
            
            if rca.should_stop():
                total_time = time.time() - start_time
                stats = rca.get_statistics()
                
                # Load best model checkpoint
                best_checkpoint_path = Path(args.checkpoint_dir) / f"best_model_epoch{stats['best_epoch']}_loss{stats['best_loss']:.6f}.pt"
                if best_checkpoint_path.exists():
                    model.load_state_dict(torch.load(best_checkpoint_path))
                    # Re-evaluate with best model
                    best_val_loss, best_val_acc = evaluate(model, device, val_loader, criterion)
                    print("\n" + "=" * 70)
                    print("🎉 TRAINING COMPLETED - EARLY STOP")
                    print("=" * 70)
                    print(f"Stopped at epoch: {epoch}/{args.epochs}")
                    print(f"Reason: RCA detected convergence")
                    saved = args.epochs - epoch
                    print(f"Epochs saved: {saved} ({saved/args.epochs*100:.1f}%)")
                    print(f"Time elapsed: {total_time:.1f}s")
                    print(f"\n✅ Best model loaded from epoch {stats['best_epoch']}")
                    print(f"Best val accuracy: {best_val_acc:.2f}% (Δ +{best_val_acc - val_acc:.2f}%)")
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
