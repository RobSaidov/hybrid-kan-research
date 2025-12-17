# -*- coding: utf-8 -*-
"""
Training Infrastructure for HybridKAN

Provides comprehensive training pipeline with:
- Mixed precision training (AMP)
- OneCycleLR scheduling with "ALL" branch policy
- Early stopping
- Gate trajectory logging
- Metric computation and confusion matrices
"""

import os
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Learning rate policy
    warmup_pct: float = 0.30
    all_branch_lr_scale: float = 0.8
    all_branch_warmup_pct: float = 0.45
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.0
    
    # Optimization
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    
    # Logging
    log_gate_every: int = 5
    save_best_only: bool = True


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    test_loss: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    learning_rate: float = 0.0
    elapsed_time: float = 0.0


class EarlyStopping:
    """Early stopping with patience and optional minimum delta."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class GateTracker:
    """Tracks gate values throughout training for analysis."""
    
    def __init__(self):
        self.branch_gate_history: List[Dict] = []
        self.residual_gate_history: List[Dict] = []
    
    def record(self, model, epoch: int):
        """Record current gate values."""
        # Branch gates
        branch_weights = model.get_branch_gate_weights()
        self.branch_gate_history.append({
            "epoch": epoch,
            "gates": branch_weights,
        })
        
        # Residual gates
        residual_weights = model.get_residual_gate_weights()
        self.residual_gate_history.append({
            "epoch": epoch,
            "gates": residual_weights,
        })
    
    def get_summary(self) -> Dict:
        """Get summary of gate evolution."""
        if not self.branch_gate_history:
            return {}
        
        initial = self.branch_gate_history[0]
        final = self.branch_gate_history[-1]
        
        return {
            "initial_branch_gates": initial,
            "final_branch_gates": final,
            "initial_residual_gates": self.residual_gate_history[0] if self.residual_gate_history else {},
            "final_residual_gates": self.residual_gate_history[-1] if self.residual_gate_history else {},
            "num_recordings": len(self.branch_gate_history),
        }
    
    def export_trajectory(self) -> Dict:
        """Export full gate trajectory for visualization."""
        return {
            "branch_gates": self.branch_gate_history,
            "residual_gates": self.residual_gate_history,
        }


class Trainer:
    """
    Comprehensive trainer for HybridKAN models.
    
    Features:
    - Mixed precision training (AMP) for GPU acceleration
    - OneCycleLR with special "ALL" branch policy
    - Early stopping
    - Gate trajectory tracking
    - Comprehensive metric logging
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainingConfig] = None,
        output_dir: str = "results",
        experiment_name: str = "hybridkan",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.metrics_history: List[TrainingMetrics] = []
        self.gate_tracker = GateTracker()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # AMP scaler
        self.use_amp = self.config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
    
    def _setup_optimizer(self):
        """Configure optimizer and learning rate scheduler."""
        # Determine if using "all" branches (affects LR policy)
        is_all_branches = len(getattr(self.model, "active_branches", [])) == 6
        
        if is_all_branches:
            max_lr = self.config.learning_rate * self.config.all_branch_lr_scale
            warmup_pct = self.config.all_branch_warmup_pct
        else:
            max_lr = self.config.learning_rate
            warmup_pct = self.config.warmup_pct
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=self.config.weight_decay,
        )
        
        steps_per_epoch = len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=self.config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=warmup_pct,
        )
        
        self.max_lr = max_lr
        self.warmup_pct = warmup_pct
    
    def _compute_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        accuracy = (y_true == y_pred).mean() * 100
        
        metrics = {"accuracy": accuracy}
        
        if SKLEARN_AVAILABLE:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            metrics.update({
                "precision": precision * 100,
                "recall": recall * 100,
                "f1": f1 * 100,
            })
        
        return metrics
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        predictions, targets = [], []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
                self.optimizer.step()
            
            self.scheduler.step()
            total_loss += loss.item()
            
            predictions.append(output.argmax(dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        
        all_preds = np.concatenate(predictions)
        all_targets = np.concatenate(targets)
        accuracy = (all_preds == all_targets).mean() * 100
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        predictions, targets = [], []
        
        for data, target in self.val_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
            else:
                output = self.model(data)
                loss = F.nll_loss(output, target)
            
            total_loss += loss.item()
            predictions.append(output.argmax(dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.concatenate(predictions)
        all_targets = np.concatenate(targets)
        
        metrics = self._compute_classification_metrics(all_targets, all_preds)
        metrics["loss"] = avg_loss
        
        return metrics
    
    def train(self, verbose: bool = True) -> Dict:
        """
        Run full training loop.
        
        Returns:
            Dict containing training summary, metrics history, and gate trajectories
        """
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="max",
        )
        
        start_time = time.time()
        
        # Record initial gate values
        self.gate_tracker.record(self.model, epoch=0)
        
        progress_bar = tqdm(
            range(1, self.config.epochs + 1),
            desc=f"Training {self.experiment_name}",
            disable=not verbose,
        )
        
        for epoch in progress_bar:
            self.current_epoch = epoch
            
            # Training
            train_loss, train_acc = self._train_epoch()
            
            # Validation
            val_metrics = self._evaluate()
            
            # Record metrics
            elapsed = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0]
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                test_loss=val_metrics["loss"],
                test_accuracy=val_metrics["accuracy"],
                test_precision=val_metrics.get("precision", 0.0),
                test_recall=val_metrics.get("recall", 0.0),
                test_f1=val_metrics.get("f1", 0.0),
                learning_rate=current_lr,
                elapsed_time=elapsed,
            )
            self.metrics_history.append(metrics)
            
            # Record gates periodically
            if epoch % self.config.log_gate_every == 0:
                self.gate_tracker.record(self.model, epoch)
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{train_loss:.4f}",
                "Train": f"{train_acc:.2f}%",
                "Val": f"{val_metrics['accuracy']:.2f}%",
                "LR": f"{current_lr:.2e}",
            })
            
            # Check for best model
            if val_metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = val_metrics["accuracy"]
                self.best_epoch = epoch
                self._save_checkpoint("best")
            
            # Early stopping
            if early_stopping(val_metrics["accuracy"]):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # Final gate recording
        self.gate_tracker.record(self.model, self.current_epoch)
        
        # Save final results
        total_time = time.time() - start_time
        summary = self._create_summary(total_time)
        self._save_results(summary)
        
        return summary
    
    def _save_checkpoint(self, tag: str = "latest"):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / "checkpoints" / f"{self.experiment_name}_{tag}.pt"
        
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_accuracy": self.best_accuracy,
            "config": asdict(self.config),
        }, checkpoint_path)
    
    def _create_summary(self, total_time: float) -> Dict:
        """Create training summary."""
        return {
            "experiment_name": self.experiment_name,
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "total_epochs": self.current_epoch,
            "total_time_seconds": total_time,
            "device": str(self.device),
            "amp_enabled": self.use_amp,
            "learning_rate_policy": {
                "max_lr": self.max_lr,
                "warmup_pct": self.warmup_pct,
            },
            "model_config": self.model.get_config() if hasattr(self.model, "get_config") else {},
            "parameter_counts": self.model.count_parameters() if hasattr(self.model, "count_parameters") else {},
            "gate_summary": self.gate_tracker.get_summary(),
        }
    
    def _save_results(self, summary: Dict):
        """Save all training results."""
        # Save summary JSON
        summary_path = self.output_dir / "logs" / f"{self.experiment_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save metrics CSV
        csv_path = self.output_dir / "logs" / f"{self.experiment_name}_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.metrics_history[0]).keys()))
            writer.writeheader()
            for metrics in self.metrics_history:
                writer.writerow(asdict(metrics))
        
        # Save gate trajectories
        gate_path = self.output_dir / "logs" / f"{self.experiment_name}_gates.json"
        with open(gate_path, "w") as f:
            json.dump(self.gate_tracker.export_trajectory(), f, indent=2, default=str)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]
    
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Compute confusion matrix on validation set."""
        if not SKLEARN_AVAILABLE:
            return None
        
        self.model.eval()
        predictions, targets = [], []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                output = self.model(data)
                predictions.append(output.argmax(dim=1).cpu().numpy())
                targets.append(target.cpu().numpy())
        
        all_preds = np.concatenate(predictions)
        all_targets = np.concatenate(targets)
        
        return confusion_matrix(all_targets, all_preds)
