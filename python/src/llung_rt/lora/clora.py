import math
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CBPLoRAGroup(nn.Module):
    """
    rank-selective LoRA modul tracks utility 
    and performs continual backprop resetting for SGD
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, replacement_rate: float = 0.05):
        super().__init__()
        self.rank = rank
        self.replacement_rate = replacement_rate
        
        # Scaling parameter (alpha / rank)
        self.scaling = 2.0 / rank
        
        # LoRA weights
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
        # Utility metric: moving average of activation magnitude
        self.register_buffer("utility", torch.zeros(rank))
        self.decay = 0.99

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                            # x shape: (batch_size, seq_len, in_dim)
        act = x @ self.A    # (batch_size, seq_len, rank)
        
        if self.training:
            with torch.no_grad():
                # mean absolute activation across batch (and sequence)
                current_util = act.abs().mean(dim=(0, 1))
                self.utility.mul_(self.decay).add_(current_util * (1 - self.decay))
                
        return (act @ self.B) * self.scaling

    def reset_low_utility_ranks(self):
        """
        re-initializes the lowest utility ranks
        pure SGD is stateless
        """
        num_resets = max(1, int(self.rank * self.replacement_rate))
        _, lowest_indices = torch.topk(self.utility, k=num_resets, largest=False)
        
        with torch.no_grad():
            for idx in lowest_indices:
                nn.init.kaiming_uniform_(self.A[:, idx : idx + 1], a=math.sqrt(5))
                # prevent sudden forward-pass output jumps
                self.B[idx : idx + 1, :].zero_()
                # reset utility
                self.utility[idx] = 0.0


class PlasticLoRATrainer:
    """
    continuous online updates on high-surprisal streams
    using SGD and warmup/restart scheduler
    """
    def __init__(self, lora_modules: list[CBPLoRAGroup], lr: float = 1e-2, T_0: int = 500):
        self.lora_modules = lora_modules
        
        # adapter parameters
        params = []
        for mod in lora_modules:
            params.extend([mod.A, mod.B])
            
        # avoid exploding gradients 
        self.optimizer = SGD(params, lr=lr, momentum=0.0, weight_decay=1e-4)
        
        # escape local minima after resetting features
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=T_0, T_mult=1, eta_min=1e-5)
        self.step_count = 0

    def train_step(self, model: nn.Module, batch: dict, reset_freq: int = 100):
        model.train()
        self.optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.step_count += 1
        
        if self.step_count % reset_freq == 0:
            for mod in self.lora_modules:
                mod.reset_low_utility_ranks()
