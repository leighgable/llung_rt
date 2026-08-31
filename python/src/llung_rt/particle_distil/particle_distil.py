import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleDistil(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.2,
        num_particles=4,
        ess_threshold=0.5,
    ):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.num_particles = num_particles
        self.ess_threshold = ess_threshold  # resampling trigger

    def compute_guided_smc_step(
        self,
        prompt_ids: torch.Tensor,
        demonstration_ids: torch.Tensor,
        max_length: int = 128,
    ):
        """
        Guided Power Sequential Monte Carlo Sampling with
        EXPECTED SEQUENCE-LEVEL ENERGY for self-distillation.
        """
        self.model.eval()

        student_particles = prompt_ids.repeat(self.num_particles, 1)
        teacher_particles = demonstration_ids.repeat(self.num_particles, 1)

        # track cumulative negative log likelihood per particle
        cumulative_nll = torch.zeros(self.num_particles, device=prompt_ids.devide)

        step_teacher_logits = []

        with torch.no_grad():
            for t in range(max_length):
                # forward pass conditioned on demo
                teacher_out = self.model(teacher_particles)
                teacher_logits = teacher_out.logits[:, -1, :] # (N, Vocab)
                step_teacher_logits.append(teacher_logits)

                # sample next token with teacher temperature
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                sampled_tokens = torch.multinomial(teacher_probs, num_samples=1)

                # accumulate negative log likelihood / sequence energy
                teacher_logp = F.log_softmax(teacher_logits, dim=-1).gather(1, sampled_tokens).squeeze(-1)
                cumulative_nll += -teacher_logp  # E_t = E_{t-1} - log p(a_t)

                student_particles = torch.cat(
                    [student_particles, sampled_tokens],
                    dim=-1,
                )
                teacher_particles = torch.cat(
                    [teacher_particles, sampled_tokens],
                    dim=-1,
                )

                # particle resampling
                seq_power_weights = F.softmax(-self.alpha * cumulative_nll, dim=0)

                ess = 1.0 / torch.sum(seq_power_weights ** 2)
                if ess < (self.ess_threshold * self.num_particles):
                    ancestors = torch.multinomial(
                        seq_power_weights,
                        num_samples=self.num_particles,
                        replacement=True,
                    )
                    student_particles = student_particles[ancestors]
                    teacher_particles = teacher_particles[ancestors]
                    cumulative_nll = cumulative_nll[ancestors]

        final_sequence_weights = F.softmax(-self.alpha * cumulative_nll, dim=0) # (N,)
        # on-policy sequence distillation
        self.model_train()

        teacher_targets = []

        for t in range(max_length):
            t_logits = step_teacher_logits[t]
            t_probs = F.softmax(t_logits, dim=-1)
            # full sequence energy weighting
            seq_weighted_target = torch.sum(
                final_sequence_weights.unsqueeze(-1) * t_probs,
                dim=0,
            )
            teacher_targets.append(seq_weighted_target)

        stacked_teacher_targets = torch.stack(teacher_targets, dim=0).unsqueeze(0)

        best_particle_idx = torch.argmax(final_sequence_weights)
        best_student_seq  = student_particles[best_particle_idx : best_particle_idx + 1]

        student_outputs = self.model(best_student_seq)
        gen_student_logits = student_outputs.logits[:, prompt_ids.size(1) - 1 : -1,:]

        # forward KL loss
        student_log_probs = F.log_softmax(gen_student_logits, dim=-1)

        return F.kl_div(
            student_log_probs,
            stacked_teacher_targets,
            reduction="batchmean",
        ) 


        

# Earlier versions

@torch.no_grad()
def compute_sequence_power_logits(
    self,
    model: nn.Module,
    prompt_and_prefix: torch.Tensor,
    candidate_logits: torch.Tensor,
    num_rollouts: int = 4,
    rollout_len: int = 32,
) -> torch.Tensor:
    """
    teacher probabilities weighted by candidate next-tokens
    by their expected future sequence-level energy:
    exp(-alpha * NLL future)
        
    """
    probs_local = F.softmax(
        candidate_logits / self.temperature, dim=-1
    )

    top_k_probs, top_k_indices = torch.topk(
                                    probs_local,
                                    k=5,
                                    dim=-1
                                )  # (B, K)

    batch_size = prompt_and_prefix.size(0)

    adjusted_logits = candidate_logits.clone()

    # estimate future energy whaaat?
    for k in range(top_k_indices.size(1)):
        candidate_token = top_k_indices[:, k : k + 1] # (B, 1) 

        # branch context
        branch_input = torch.cat([prompt_and_prefix, candidate_token], dim=-1)

        # MCC
        total_nll = torch.zeros(batch_size, device=branch_input.device)

        for _ in range(num_rollouts):
            rollout_ids = branch_input.clone()
            rollout_nll = 0.0

            for step in range(rollout_len):
                out = model(rollout_ids)
                next_logits = out.logits[:, -1, :] / self.temperature
                log_p = F.log_softmax(next_logits, dim=-1)

                # sample continuation
                next_tok = torch.multinomial(torch.exp(log_p), num_samples=1)

                # -log p(y_step)
                step_nll = -log_p.gather(dim=-1, index=next_tok).squeeze(-1)
                rollout_nll += step_nll

                rollout_ids = torch.cat([rollout_ids, next_tok], dim=-1)

            total_nll += rollout_nll / rollout_len

        avg_future_nll = total_nll / num_rollouts

        # power reweighting exp(-alpha * NLL)
        power_bonus = -self.alpha * avg_future_nll

        # adjust candidate logit
        adjusted_logits.scatter_(
            dim=-1,
            index=candidate_token,
            src=(candidate_logits.gather(
                    -1,
                    candidate_token
                ) + power_bonus.unsqueeze(-1)
            )
        )

    return adjusted_logits

@torch.no_grad()
def compute_power_logits(
    logits: torch.Tensor,
    temperature: torch.float32,
    alpha: torch.float32,
) -> torch.Tensor:
    """
    power sampling to raw logits, equivalent to exponentiating probs
    """
    return logits / (temperature / alpha)
