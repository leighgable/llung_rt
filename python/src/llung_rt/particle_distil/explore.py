import torch
import torch.nn.functional as F

def compute_surprisal(
    teacher_logits: list[torch.Tensor],
    student_logits: list[torch.Tensor],
) -> torch.Tensor:
    """
    trajectory-level surprisal ie. the student finds
    teacher's choices surprising relative to its prior
    """
    total_surprisal = 0.0
    for t_logits, s_logits in zip(teacher_logits, student_logits):
        # KL divergence per token
        t_logp = F.log_softmax(t_logits, dim=-1)
        s_logp = F.log_softmax(s_logits, dim=-1)
        # pointwise surprisal
        total_surprisal += torch.sum(s_logp.exp() * (s_logp - t_logp), dim=-1)
    return total_surprisal

def aggregate_targets(
    step_teacher_logits: list[torch.Tensor],
    cumulative_nll: torch.Tensor,
    surprisal: torch.Tensor,
    alpha: float,
    beta: float,
    use_winner_take_all: bool,
) -> torch.Tensor:
    """ target generator for step logits """

    combined_energy = (-alpha * cumulative_nll) + (beta * surprisal)
    weights = F.softmax(combined_energy, dim=0) # (N,)

    # 2. Build target tensor across steps
    targets = []
    for t_logits in step_teacher_logits:
        t_probs = F.softmax(t_logits, dim=-1)
        if use_winner_take_all:
            best_idx = torch.argmax(weights)
            target_t = t_probs[best_idx]
        else:
            target_t = torch.sum(weights.unsqueeze(-1) * t_probs, dim=0)
            
        targets.append(target_t)

    return torch.stack(targets, dim=0).unsqueeze(0), weights
                                                    # (1, max_length, Vocab)
