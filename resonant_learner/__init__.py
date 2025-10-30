"""
Resonant Learner - Community Edition

Intelligent early stopping for neural networks using log-periodic resonance analysis.

Example:
    >>> from resonant_learner import ResonantCallback
    >>> 
    >>> rca = ResonantCallback(checkpoint_dir='./checkpoints')
    >>> 
    >>> for epoch in range(max_epochs):
    ...     train_loss = train_epoch(model, train_loader, optimizer)
    ...     val_loss = validate(model, val_loader)
    ...     rca(val_loss=val_loss, model=model, optimizer=optimizer)
    ...     if rca.should_stop():
    ...         break
"""

__version__ = "1.0.0"
__author__ = "Damjan Žakelj"
__license__ = "MIT"

from .resonant_callback import ResonantCallback

__all__ = ['ResonantCallback']