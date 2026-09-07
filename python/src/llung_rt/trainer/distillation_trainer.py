import os
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

class MonitoredDistillationTrainer:
    def __init__(
        self,
        distiller_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: str = "./runs/self_distillation",
        checkpoint_dir: str = "./checkpoints",
        max_nan_streak: int = 3,
        stagnation_patience: int = 100,
        min_loss_delta: float = 1e-4,
    ):
        self.model = distiller_model
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Health & Monitoring Config
        self.max_nan_streak = max_nan_streak
        self.stagnation_patience = stagnation_patience
        self.min_loss_delta = min_loss_delta

        # State Trackers
        self.nan_streak = 0
        self.best_loss = float("inf")
        self.recent_losses = []
        self.stagnant_steps = 0

    def check_health_and_step(
        self,
        loss: torch.Tensor,
        current_step: int,
        extra_metrics: Optional[dict[str, float]] = None
    ) -> bool:
        """
        Validates loss/gradients, logs to TensorBoard, saves checkpoints, 
        and signals whether training should stop early.
        Returns: True to continue, False to abort training.
        """
        loss_val = loss.item()

        # --- 1. NaN / Inf Detection ---
        if math.isnan(loss_val) or math.isinf(loss_val):
            self.nan_streak += 1
            print(f"⚠️ [Step {current_step}] NaN/Inf loss detected! (Streak: {self.nan_streak}/{self.max_nan_streak})")
            self.optimizer.zero_grad() # Flush bad gradients
            
            if self.nan_streak >= self.max_nan_streak:
                print("NaNs detected. Terminating run.")
                return False
            return True # Skip step, attempt recovery on next batch

        # Reset NaN counter on healthy step
        self.nan_streak = 0

        # --- 2. Gradient Clipping & Norm Logging ---
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0).item()
        
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            print(f"[Step {current_step}] NaN gradient norm detected! Skipping optimizer step.")
            self.optimizer.zero_grad()
            return True

        self.optimizer.step()
        self.optimizer.zero_grad()

        # --- 3. TensorBoard Logging ---
        self.writer.add_scalar("Train/Loss", loss_val, current_step)
        self.writer.add_scalar("Train/GradNorm", grad_norm, current_step)

        if extra_metrics:
            for metric_name, val in extra_metrics.items():
                self.writer.add_scalar(metric_name, val, current_step)

        # --- 4. Stagnation / Stopped Learning Guard ---
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 50:
            self.recent_losses.pop(0)

        moving_avg_loss = sum(self.recent_losses) / len(self.recent_losses)
        self.writer.add_scalar("Train/Loss_MovingAvg", moving_avg_loss, current_step)

        if len(self.recent_losses) >= 50:
            # Check if loss improvement is stalled
            if (self.best_loss - moving_avg_loss) < self.min_loss_delta:
                self.stagnant_steps += 1
            else:
                self.stagnant_steps = 0

            if self.stagnant_steps >= self.stagnation_patience:
                print(f" [Step {current_step}] Learning stagnated for {self.stagnant_steps} steps. Early stopping.")
                return False

        # --- 5. Auto-Checkpointing ---
        if moving_avg_loss < self.best_loss:
            self.best_loss = moving_avg_loss
            self.save_checkpoint("checkpoint_best.pt", current_step, moving_avg_loss)

        if current_step % 200 == 0:
            self.save_checkpoint("checkpoint_latest.pt", current_step, loss_val)

        return True

    def save_checkpoint(self, filename: str, step: int, loss: float):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "best_loss": self.best_loss,
        }, path)

    def close(self):
        self.writer.close()
