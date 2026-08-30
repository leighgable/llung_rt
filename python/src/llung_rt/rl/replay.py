import torch
import torch.nn as nn
from collections import deque
import random
from typing import Dict, List, Optional

class ReplayBufferGRPO:
    """
    historical rollouts alongside log probabilities of policy 
    """
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, rollout: Dict):
        """
        'prompt_ids': Tensor
        'completion_ids': Tensor
        'reward': float
        'behavior_logps': Tensor (CPU) -> log prob under generation policy
        'ref_logps': Tensor (CPU)      -> log prob under frozen reference model
        """
        self.buffer.append(rollout)

    def sample_group(self, group_size: int = 32) -> Optional[List[Dict]]:
        if len(self.buffer) < group_size:
            return None
        return random.sample(list(self.buffer), group_size)


class OffPolicyStreamingGRPOTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        lora_modules: list,
        optimizer: torch.optim.Optimizer,
        buffer: ReplayBufferGRPO,
        beta: float = 0.04,        # KL Penalty
        clip_eps: float = 0.2,     # PPO clip parameter
        max_is_weight: float = 5.0 # Upper bound truncation for IS weights
    ):
        self.model = model
        self.lora_modules = lora_modules
        self.optimizer = optimizer
        self.buffer = buffer
        self.beta = beta
        self.clip_eps = clip_eps
        self.max_is_weight = max_is_weight

    def train_step_from_buffer(
        self, 
        group_size: int = 32, 
        prompt_len: int = 64
    ) -> Optional[float]:
        """
        streaming GRPO update with historical rollouts 
        """
        rollouts = self.buffer.sample_group(group_size)
        if rollouts is None:
            return None  # Not enough samples in buffer yet

        rewards = torch.tensor([r['reward'] for r in rollouts], dtype=torch.float32)
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0

        for i, rollout in enumerate(rollouts):
            # Move individual sequence to GPU
            device = next(self.model.parameters()).device
            completion_ids = rollout['completion_ids'].to(device)
            full_input_ids = torch.cat([rollout['prompt_ids'].to(device), completion_ids], dim=-1)
            
            behavior_logps = rollout['behavior_logps'].to(device)
            ref_logps = rollout['ref_logps'].to(device)
            A_i = advantages[i].to(device)

            logits = self.model(full_input_ids).logits[:, prompt_len - 1 : -1, :]
            policy_log_probs = torch.log_softmax(logits, dim=-1)
            active_logps = torch.gather(policy_log_probs, -1, completion_ids.unsqueeze(-1)).squeeze(-1)

            # ratio = pi_current(x) / pi_behavior(x)
            log_ratio = active_logps - behavior_logps
            ratio = torch.exp(log_ratio)

            # truncate high-variance Importance Sampling weights
            truncated_ratio = torch.clamp(ratio, max=self.max_is_weight)

            surr1 = truncated_ratio * A_i
            surr2 = torch.clamp(truncated_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * A_i
            policy_loss = -torch.min(surr1, surr2).mean()

            kl_div = torch.exp(ref_logps - active_logps) - (ref_logps - active_logps) - 1
            kl_loss = self.beta * kl_div.mean()

            # normalize loss 
            sample_loss = (policy_loss + kl_loss) / group_size
            sample_loss.backward()

            total_loss += sample_loss.item() * group_size

            del logits, policy_log_probs, active_logps, sample_loss

        # Step Optimizer
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        return total_loss
