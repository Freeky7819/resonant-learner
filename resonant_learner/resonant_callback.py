#!/usr/bin/env python3
"""
resonant_callback.py - Resonant Convergence Analysis (RCA) Callback

Community Edition - Intelligent early stopping using log-periodic resonance analysis.

Author: Damjan Žakelj
License: MIT
"""

from __future__ import annotations
import math
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim


class ResonantCallback:
    """
    Intelligent early stopping using Resonant Convergence Analysis (RCA).
    
    RCA analyzes the oscillation patterns in validation loss to detect true convergence
    versus temporary plateaus, enabling earlier and more accurate stopping decisions.
    
    Key Features:
    - Log-periodic resonance analysis of validation loss
    - Adaptive learning rate scheduling
    - Automatic checkpoint saving
    - Overfitting detection
    
    Args:
        checkpoint_dir: Directory to save best model checkpoints
        patience_steps: Number of epochs to wait before reducing LR
        min_delta: Minimum improvement threshold (relative)
        ema_alpha: EMA smoothing factor for loss tracking
        max_lr_reductions: Maximum number of LR reductions before stopping
        lr_reduction_factor: Factor to reduce LR by (default 0.5)
        min_lr: Minimum learning rate threshold
        verbose: Print RCA analysis messages
        enable_tensorboard: Enable TensorBoard logging (not implemented in Community Edition)
    
    Example:
        >>> rca = ResonantCallback(checkpoint_dir='./checkpoints')
        >>> for epoch in range(max_epochs):
        ...     train_loss = train_epoch(model, train_loader, optimizer)
        ...     val_loss = validate(model, val_loader)
        ...     rca(val_loss=val_loss, model=model, optimizer=optimizer)
        ...     if rca.should_stop():
        ...         break
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        patience_steps: int = 3,
        min_delta: float = 0.01,
        ema_alpha: float = 0.3,
        max_lr_reductions: int = 2,
        lr_reduction_factor: float = 0.5,
        min_lr: float = 1e-6,
        verbose: bool = True,
        enable_tensorboard: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.patience_steps = patience_steps
        self.min_delta = min_delta
        self.ema_alpha = ema_alpha
        self.max_lr_reductions = max_lr_reductions
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.verbose = verbose
        
        if enable_tensorboard:
            warnings.warn("TensorBoard logging not available in Community Edition")
        
        # State tracking
        self.loss_history = []
        self.loss_ema = None
        self.prev_loss_ema = None
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.epochs_since_improvement = 0
        self.lr_reductions = 0
        self._should_stop_flag = False
        self.current_epoch = 0
    
    def _update_ema(self, loss: float) -> float:
        """Update EMA of validation loss and return delta."""
        if self.loss_ema is None:
            self.loss_ema = loss
            self.prev_loss_ema = loss
            return 0.0
        
        self.prev_loss_ema = self.loss_ema
        self.loss_ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.loss_ema
        return self.loss_ema - self.prev_loss_ema
    
    def _compute_rca_metrics(self) -> Dict[str, float]:
        """
        Compute RCA resonance metrics: beta (amplitude) and omega (frequency).
        
        Uses log-periodic analysis of recent loss history to detect convergence patterns.
        """
        if len(self.loss_history) < 4:
            return {'beta': 0.0, 'omega': 0.0, 'confidence': 0.0}
        
        # Use recent window for analysis
        window = self.loss_history[-10:]
        
        # Compute oscillation amplitude (beta)
        if len(window) >= 2:
            diffs = [abs(window[i] - window[i-1]) for i in range(1, len(window))]
            max_diff = max(diffs) if diffs else 0.0
            avg_loss = sum(window) / len(window)
            beta = 1.0 - (max_diff / (avg_loss + 1e-8))  # Normalized amplitude
            beta = max(0.0, min(1.0, beta))  # Clamp to [0, 1]
        else:
            beta = 0.0
        
        # Compute resonance frequency (omega)
        # Count zero-crossings in derivative
        if len(window) >= 3:
            derivs = [window[i] - window[i-1] for i in range(1, len(window))]
            sign_changes = sum(1 for i in range(1, len(derivs)) 
                             if derivs[i] * derivs[i-1] < 0)
            # Map to frequency around 6.0 (universal resonance frequency)
            omega = 2.0 * math.pi * (sign_changes / max(len(derivs), 1))  # Natural frequency
            omega = max(1.0, min(12.0, omega))
        else:
            omega = 0.0
        
        # Compute confidence based on history length and stability
        confidence = min(1.0, len(self.loss_history) / 10.0)
        if len(window) >= 5:
            recent_std = math.sqrt(sum((x - sum(window)/len(window))**2 for x in window) / len(window))
            stability = 1.0 / (1.0 + recent_std)
            confidence *= stability
        
        return {
            'beta': beta,
            'omega': omega,
            'confidence': confidence
        }
    
    def _get_state(self, beta: float, omega: float) -> str:
        """Determine convergence state from RCA metrics."""
        if beta > 0.85 and 5.5 <= omega <= 6.5:
            return "converging"
        elif beta > 0.7:
            return "plateau"
        elif omega > 8.0 or beta < 0.3:
            return "unstable"
        else:
            return "improving"
    
    def _should_reduce_lr(self) -> bool:
        """Check if learning rate should be reduced."""
        return self.epochs_since_improvement >= self.patience_steps
    
    def _reduce_lr(self, optimizer: optim.Optimizer) -> bool:
        """Reduce learning rate if not already at minimum."""
        if self.lr_reductions >= self.max_lr_reductions:
            return False
        
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.lr_reduction_factor, self.min_lr)
            
            if new_lr >= self.min_lr and new_lr < old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"🔻 RCA: Reducing LR: {old_lr:.6f} → {new_lr:.6f}")
                self.lr_reductions += 1
                self.epochs_since_improvement = 0
                return True
        
        return False
    
    def _save_checkpoint(self, model: nn.Module, val_loss: float):
        """Save model checkpoint if it's the best so far."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.current_epoch
            
            checkpoint_path = self.checkpoint_dir / f"best_model_epoch{self.current_epoch}_loss{val_loss:.6f}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            if self.verbose:
                print(f"💾 Checkpoint saved: {checkpoint_path.name}")
    
    def __call__(
        self,
        val_loss: float,
        model: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        epoch: Optional[int] = None
    ):
        """
        Main callback - call after each validation epoch.
        
        Args:
            val_loss: Validation loss for current epoch
            model: Model (for checkpoint saving)
            optimizer: Optimizer (for LR adjustment)
            epoch: Current epoch number (optional, auto-incremented if None)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        # Update loss tracking
        self.loss_history.append(val_loss)
        delta = self._update_ema(val_loss)
        
        # Compute RCA metrics
        metrics = self._compute_rca_metrics()
        beta = metrics['beta']
        omega = metrics['omega']
        confidence = metrics['confidence']
        state = self._get_state(beta, omega)
        
        # Check for improvement
        if self.best_loss == float('inf'):
            # First epoch - always consider it an improvement
            improved = True
            relative_improvement = 1.0
        else:
            relative_improvement = (self.best_loss - val_loss) / self.best_loss
            improved = relative_improvement > self.min_delta
        
        if improved:
            self.epochs_since_improvement = 0
            prev_best = self.best_loss  # Save before update
            if model is not None:
                self._save_checkpoint(model, val_loss)
            
            if self.verbose:
                print(f"📊 RCA (Epoch {self.current_epoch}): Improvement! Val Loss: {val_loss:.6f} (prev: {prev_best:.6f})")
                print(f"  β={beta:.2f}, ω={omega:.1f}, confidence={confidence:.2f}, state={state}")
        else:
            self.epochs_since_improvement += 1
            
            if self.verbose:
                print(f"📊 RCA (Epoch {self.current_epoch}): No improvement (waiting {self.epochs_since_improvement}/{self.patience_steps})")
                print(f"  β={beta:.2f}, ω={omega:.1f}, confidence={confidence:.2f}, state={state}")
        
        # Check if should reduce LR
        if optimizer is not None and self._should_reduce_lr():
            lr_reduced = self._reduce_lr(optimizer)
            
            # If max reductions reached and still converging, stop
            if not lr_reduced and state == "converging":
                if self.verbose:
                    print(f"🛑 RCA: Early stopping triggered!")
                    print(f"  Reason: Strong convergence signal (β={beta:.2f}, ω={omega:.1f})")
                    print(f"  Best model saved at epoch {self.best_epoch} (val_loss={self.best_loss:.6f})")
                self._should_stop_flag = True
            # 🔥 NEW: If max reductions reached and still no improvement, stop
            elif not lr_reduced and self.epochs_since_improvement >= self.patience_steps:
                if self.verbose:
                    print(f"🛑 RCA: Early stopping triggered!")
                    print(f"  Reason: Max LR reductions reached, no improvement for {self.epochs_since_improvement} epochs")
                    print(f"  Best model saved at epoch {self.best_epoch} (val_loss={self.best_loss:.6f})")
                self._should_stop_flag = True
        
        # Check for plateau - independent of LR reduction status
        # More aggressive: stop if beta is very high even before full patience
        if state == "plateau" and beta > 0.85 and self.epochs_since_improvement >= 2:
            if self.verbose:
                print(f"🛑 RCA: Early stopping triggered!")
                print(f"  Reason: Strong plateau detected (β={beta:.2f}, no improvement for {self.epochs_since_improvement} epochs)")
                print(f"  Best model saved at epoch {self.best_epoch} (val_loss={self.best_loss:.6f})")
            self._should_stop_flag = True
        elif state == "plateau" and beta > 0.80 and self.epochs_since_improvement >= self.patience_steps:
            if self.verbose:
                print(f"🛑 RCA: Early stopping triggered!")
                print(f"  Reason: Stable plateau detected (β={beta:.2f}, no improvement for {self.epochs_since_improvement} epochs)")
                print(f"  Best model saved at epoch {self.best_epoch} (val_loss={self.best_loss:.6f})")
            self._should_stop_flag = True
        
        # 🔥 NEW: Overfitting detection - val_loss increasing consistently
        if len(self.loss_history) >= 3:
            # Check if last 3 losses are increasing (overfitting signal)
            recent_losses = self.loss_history[-3:]
            is_increasing = all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1))
            
            if is_increasing and self.epochs_since_improvement >= 2:
                if self.verbose:
                    print(f"🛑 RCA: Early stopping triggered!")
                    print(f"  Reason: Overfitting detected (val_loss increasing for 3 epochs: {recent_losses[0]:.4f} → {recent_losses[-1]:.4f})")
                    print(f"  Best model saved at epoch {self.best_epoch} (val_loss={self.best_loss:.6f})")
                self._should_stop_flag = True
        
        # Strong convergence signal - universal resonance frequency
        # Early stop if approaching universal frequency with high beta
        if beta > 0.85 and 4.5 <= omega <= 6.5 and confidence > 0.7:
            if self.verbose:
                print(f"🛑 RCA: Early stopping triggered!")
                print(f"  Reason: Approaching universal resonance (β={beta:.2f}, ω={omega:.1f})")
                print(f"  Best model saved at epoch {self.best_epoch} (val_loss={self.best_loss:.6f})")
            self._should_stop_flag = True
        elif beta > 0.90 and 5.8 <= omega <= 6.2 and confidence > 0.8:
            if self.verbose:
                print(f"🛑 RCA: Early stopping triggered!")
                print(f"  Reason: Strong convergence signal at universal resonance (β={beta:.2f}, ω={omega:.1f})")
                print(f"  Best model saved at epoch {self.best_epoch} (val_loss={self.best_loss:.6f})")
            self._should_stop_flag = True
        
        # Return stop signal for convenience (same as should_stop())
        return self._should_stop_flag
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self._should_stop_flag
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RCA statistics."""
        metrics = self._compute_rca_metrics()
        return {
            'current_epoch': self.current_epoch,
            'best_epoch': self.best_epoch,
            'best_loss': self.best_loss,
            'lr_reductions': self.lr_reductions,
            'beta': metrics['beta'],
            'omega': metrics['omega'],
            'confidence': metrics['confidence'],
            'state': self._get_state(metrics['beta'], metrics['omega']),
        }
