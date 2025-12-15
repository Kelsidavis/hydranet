"""Training loop for HydraNet on consumer hardware."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import time
import json
import math

# ========== RESOURCE LIMITS ==========
# Reserve 2 CPU threads and 6GB RAM for system stability
RESERVED_THREADS = 2
RESERVED_RAM_GB = 6
MAX_CPU_THREADS = max(1, os.cpu_count() - RESERVED_THREADS) if os.cpu_count() else 4
MAX_DATALOADER_WORKERS = max(1, MAX_CPU_THREADS // 4)  # Conservative worker count

# Limit PyTorch CPU threads
torch.set_num_threads(MAX_CPU_THREADS)
torch.set_num_interop_threads(max(1, MAX_CPU_THREADS // 2))

# Limit OMP/MKL threads
os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_CPU_THREADS)

# Calculate available RAM (for config sizing, no hard limits)
TOTAL_RAM_GB = 128  # fallback
AVAILABLE_RAM_GB = TOTAL_RAM_GB - RESERVED_RAM_GB
try:
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemTotal:'):
                total_kb = int(line.split()[1])
                TOTAL_RAM_GB = total_kb / (1024 * 1024)
                AVAILABLE_RAM_GB = TOTAL_RAM_GB - RESERVED_RAM_GB
                break
except Exception:
    pass
# =====================================

from ..model.config import HydraNetConfig
from ..model.hydranet import HydraNet


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Batch sizes
    batch_size: int = 1  # Per-device batch size
    gradient_accumulation_steps: int = 16  # Effective batch = batch_size * grad_accum

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True  # Use fp16/bf16
    compile_model: bool = False  # torch.compile (requires PyTorch 2.0+)

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "./checkpoints"

    # Logging
    log_steps: int = 10
    wandb_project: Optional[str] = None

    # Optimizer
    optimizer: str = "adamw"  # adamw, adam8bit, lion
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # constant, linear, cosine
    min_lr_ratio: float = 0.1

    # MoE specific
    aux_loss_weight: float = 0.01
    load_balance_loss: bool = True


class CosineScheduler:
    """Cosine learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr_mult = self._get_lr_mult()

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult

    def _get_lr_mult(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay


class HydraNetTrainer:
    """
    Trainer for HydraNet optimized for consumer hardware.

    Features:
    - Gradient accumulation for effective large batch
    - Mixed precision training (fp16/bf16)
    - Gradient checkpointing
    - MoE auxiliary loss handling
    - Memory-efficient optimizer options
    """

    def __init__(
        self,
        model: HydraNet,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn or self._default_collate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Compile model (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')

        # Logging
        self.log_history = []

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay handling."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_eps,
            )
        elif self.config.optimizer == "adam8bit":
            try:
                import bitsandbytes as bnb
                return bnb.optim.Adam8bit(
                    param_groups,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                )
            except ImportError:
                print("bitsandbytes not installed, falling back to AdamW")
                return torch.optim.AdamW(param_groups, lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _default_collate(self, batch):
        """Default collate function for language modeling."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'labels': labels}

    def train(self):
        """Main training loop."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=MAX_DATALOADER_WORKERS,
            pin_memory=True,
        )

        self.model.train()
        accumulated_loss = 0.0
        accumulated_aux_loss = 0.0
        num_accumulated = 0

        start_time = time.time()

        while self.global_step < self.config.max_steps:
            for batch in train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass with mixed precision
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(input_ids)
                    logits = outputs.logits

                    # Language modeling loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                    # Add MoE auxiliary loss
                    if self.config.load_balance_loss and outputs.aux_loss is not None:
                        aux_loss = outputs.aux_loss * self.config.aux_loss_weight
                        total_loss = loss + aux_loss
                    else:
                        aux_loss = torch.tensor(0.0)
                        total_loss = loss

                    # Scale for gradient accumulation
                    total_loss = total_loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                accumulated_loss += loss.item()
                accumulated_aux_loss += aux_loss.item()
                num_accumulated += 1

                # Gradient accumulation step
                if num_accumulated >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.log_steps == 0:
                        avg_loss = accumulated_loss / num_accumulated
                        avg_aux = accumulated_aux_loss / num_accumulated
                        elapsed = time.time() - start_time
                        tokens_per_sec = (
                            self.global_step *
                            self.config.batch_size *
                            self.config.gradient_accumulation_steps *
                            input_ids.shape[1]
                        ) / elapsed

                        log_entry = {
                            'step': self.global_step,
                            'loss': avg_loss,
                            'aux_loss': avg_aux,
                            'lr': self.optimizer.param_groups[0]['lr'],
                            'grad_norm': grad_norm.item(),
                            'tokens_per_sec': tokens_per_sec,
                        }
                        self.log_history.append(log_entry)

                        print(
                            f"Step {self.global_step}: "
                            f"loss={avg_loss:.4f} "
                            f"aux={avg_aux:.4f} "
                            f"lr={log_entry['lr']:.2e} "
                            f"tok/s={tokens_per_sec:.0f}"
                        )

                    # Evaluation
                    if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        print(f"Eval loss: {eval_loss:.4f}")

                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint("best")

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")

                    # Reset accumulation
                    accumulated_loss = 0.0
                    accumulated_aux_loss = 0.0
                    num_accumulated = 0

                    # Check if done
                    if self.global_step >= self.config.max_steps:
                        break

            self.epoch += 1

        # Final save
        self.save_checkpoint("final")

        return self.log_history

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate model on eval dataset."""
        if self.eval_dataset is None:
            return float('inf')

        self.model.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=max(1, MAX_DATALOADER_WORKERS // 2),
        )

        total_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(input_ids)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            total_loss += loss.item()
            num_batches += 1

            # Limit eval batches
            if num_batches >= 100:
                break

        self.model.train()
        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save optimizer state
        torch.save(
            self.optimizer.state_dict(),
            checkpoint_dir / "optimizer.pt",
        )

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'config': self.config.__dict__,
        }
        with open(checkpoint_dir / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(path)

        # Load optimizer
        opt_path = checkpoint_dir / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(torch.load(opt_path))

        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_eval_loss = state['best_eval_loss']
            self.scheduler.current_step = self.global_step


class SimpleTextDataset(Dataset):
    """Simple text dataset for testing."""

    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
        }


def estimate_training_time(
    model: HydraNet,
    config: TrainingConfig,
    dataset_size: int,
    tokens_per_second: float = 10000,  # Estimated throughput
) -> Dict[str, Any]:
    """Estimate training time and resources."""

    total_tokens = dataset_size * config.max_steps
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch

    total_hours = total_tokens / tokens_per_second / 3600

    # Memory estimate
    params = model.num_parameters()

    # Mixed precision: params * 2 (fp16) + optimizer states * 8 (fp32 master + momentum + variance)
    param_memory_gb = params * 2 / 1e9  # Model in fp16
    optimizer_memory_gb = params * 12 / 1e9  # Adam states

    # Activation memory (rough estimate)
    activation_gb = 2.0  # With gradient checkpointing

    total_memory_gb = param_memory_gb + optimizer_memory_gb + activation_gb

    return {
        'total_parameters': params,
        'total_parameters_billions': params / 1e9,
        'estimated_hours': total_hours,
        'estimated_days': total_hours / 24,
        'steps_per_epoch': steps_per_epoch,
        'total_epochs': config.max_steps / steps_per_epoch,
        'param_memory_gb': param_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'total_memory_gb': total_memory_gb,
        'fits_in_16gb': total_memory_gb < 14,  # Leave headroom
    }
