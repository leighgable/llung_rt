import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, PeftModel
import typer

app = typer.Typer(help="Continual Backprop SFT Trainer for LoRA Adapters")


class CBPLoRALayer(nn.Module):
    """
    continual backpropagation, utility tracking, feat resetting, SGD optimization.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, replacement_rate: float = 0.05):
        super().__init__()
        self.rank = rank
        self.replacement_rate = replacement_rate
        self.scaling = 2.0 / rank

        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # exponential moving average f rank activations
        self.register_buffer("utility", torch.zeros(rank))
        self.decay = 0.99

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = x @ self.A  # (batch, seq, rank)
        if self.training:
            with torch.no_grad():
                current_util = act.abs().mean(dim=(0, 1))
                self.utility.mul_(self.decay).add_(current_util * (1 - self.decay))
        return (act @ self.B) * self.scaling

    def reset_low_utility_ranks(self):
        """lowest utility ranks for Continual Backprop"""
        num_resets = max(1, int(self.rank * self.replacement_rate))
        _, lowest_indices = torch.topk(self.utility, k=num_resets, largest=False)

        with torch.no_grad():
            for idx in lowest_indices:
                nn.init.kaiming_uniform_(self.A[:, idx : idx + 1], a=math.sqrt(5))
                self.B[idx : idx + 1, :].zero_()
                self.utility[idx] = 0.0


class CBPSFTTrainer:
    """
    initial LoRA training via SGD, cosine annealing with warm restarts,
    and periodic rank rest
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        cbp_layers: list[CBPLoRALayer],
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        t_0: int = 200,
        reset_freq: int = 50,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cbp_layers = cbp_layers
        self.reset_freq = reset_freq
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t_0, T_mult=1, eta_min=1e-5
        )

    def train_epoch(self, dataloader: DataLoader, device: torch.device) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=self.max_grad_norm,
            )

            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            if self.global_step % self.reset_freq == 0:
                for layer in self.cbp_layers:
                    layer.reset_low_utility_ranks()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(dataloader)
        perplexity = math.exp(avg_loss) if avg_loss < 30 else float("inf")
        return {"eval_loss": avg_loss, "perplexity": perplexity}

    def save_adapter(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"LoRA adapters successfully saved to: {output_dir}")


def setup_model_and_dataset(
    model_name: str, lora_r: int, lora_alpha: int
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    return model, tokenizer


def dummy_text_dataset(tokenizer, num_samples: int = 100, seq_len: int = 128):
    data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,))
        data.append({"input_ids": input_ids, "labels": input_ids.clone()})
    return data


# ----------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------

@app.command()
def train(
    model_name: str = typer.Option("gpt2", help="Base HF Model path/name"),
    output_dir: str = typer.Option("./cbp_lora_output", help="Save directory for adapters"),
    epochs: int = typer.Option(3, help="Number of SFT epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    lr: float = typer.Option(1e-2, help="Learning rate for SGD"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha scaling factor"),
    reset_freq: int = typer.Option(50, help="Step frequency for CBP resets"),
):
    """Run SFT with Continual Backprop (SGD + LR Scheduler + Utility Resets)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model '{model_name}' onto device: {device}")

    model, tokenizer = setup_model_and_dataset(model_name, lora_r, lora_alpha)
    model.to(device)

    # Wrap target LoRA matrices for CBP tracking
    cbp_layers = []
    for name, module in model.named_modules():
        if "lora_B" in name and isinstance(module, nn.Linear):
            # Track LoRA ranks
            cbp_layer = CBPLoRALayer(module.in_features, module.out_features, rank=lora_r)
            cbp_layer.to(device)
            cbp_layers.append(cbp_layer)

    dataset = dummy_text_dataset(tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collator_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)
    )

    trainer = CBPSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        cbp_layers=cbp_layers,
        lr=lr,
        reset_freq=reset_freq,
    )

    print("\nStarting Continual Backpropagation SFT Phase...")
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(dataloader, device)
        eval_res = trainer.evaluate(dataloader, device)
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {loss:.4f} | "
            f"Eval Loss: {eval_res['eval_loss']:.4f} | Perplexity: {eval_res['perplexity']:.2f}"
        )

    trainer.save_adapter(output_dir)


@app.command()
def eval(
    model_name: str = typer.Option("gpt2", help="Base HF Model path/name"),
    adapter_dir: str = typer.Option("./cbp_lora_output", help="Path to saved LoRA adapter"),
    batch_size: int = typer.Option(4, help="Batch size"),
):
    """Evaluate trained LoRA adapter on evaluation dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating adapter from '{adapter_dir}' on base model '{model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)

    dataset = dummy_text_dataset(tokenizer, num_samples=20)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Use standard evaluation
    trainer = CBPSFTTrainer(model=model, tokenizer=tokenizer, cbp_layers=[])
    metrics = trainer.evaluate(dataloader, device)

    print(f"\nResults: Eval Loss = {metrics['eval_loss']:.4f} | Perplexity = {metrics['perplexity']:.2f}")


if __name__ == "__main__":
    app()
