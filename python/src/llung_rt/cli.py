import os
import yaml
import contextlib
import datetime
import click
import typer
from pathlib import Path
from typing import Optional
import torch

from utils.utils import load_model_and_tokenizer, stream_distillation_dataset
from particle_distil.particle_distil import ParticleDistil
from trainer.distillation_trainer import MonitoredDistillationTrainer

app = typer.Typer(
    name="sdft",
    help="self-distillation with particle sampling",
    no_args_is_help=True,
)

@app.command()
def particle_trainer(
    config_path: Optional[Path] = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to a YAML configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(
        True,
        help="whether to print progress logs."
    ),
):
    """
    launch particle self-distillation training pipeline
    """

    if not os.path.exists(config_path):
        typer.echo(f"Error: Configuration file not found at {config_path}", err=True)
        raise typer.Exit(code=1)

    # 1. Load the YAML configuration
    if verbose:
        typer.echo(f"Loading configuration from: {config_path}")
        
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            typer.echo(f"Error parsing YAML file: {exc}", err=True)
            raise typer.Exit(code=1)
        
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})

    device = model_cfg.get("device", {})

    dataset = stream_distillation_dataset(
        config=data_cfg,
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_cfg.get("model_name_or_path", {}),
        device=device,
    )

    distiller = ParticleDistil(
        model=model,
        alpha=model_cfg.get("alpha", 1.2),
        beta_start=model_cfg.get("beta_start", 1.0),
        total_steps=model_cfg.get("total_steps", 0),
        num_particles=model_cfg.get("num_particles", 4),
        ess_threshold=model_cfg.get("ess_threshold", 0.5),
    )
    optimizer = torch.optim.AdamW(distiller.model.parameters(), lr=1e-5)    

    
    timestamp = Path(datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    ))
    log_dir = Path(trainer_cfg.get("log_dir", "")).joinpath(timestamp)
    checkpoint_dir = Path(
        trainer_cfg.get("checkpoint_dir", "")
    ).joinpath(timestamp)    

    trainer = MonitoredDistillationTrainer(
        distiller_model=distiller,
        optimizer=optimizer,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_nan_streak=trainer_cfg.get("max_nan_streak", 10),
        stagnation_patience=trainer_cfg.get("stagnation_patience", 150),
        min_loss_delta=trainer_cfg.get("min_loss_delta"),
    )

    for step, batch in enumerate(dataset):
        prompt_ids = batch["prompt_ids"].to(device)
        demo_ids = batch["demonstration_ids"].to(device)

        # forward!
        loss, smc_metrics = distiller.guided_smc_step(
            prompt_ids=prompt_ids,
            demonstration_ids=demo_ids,
            current_step=step,
            max_length=model_cfg.get("max_length", 512),
        )

        loss.backward()

        should_continue = trainer.check_health_and_step(
            loss=loss,
            current_step=step,
            extra_metrics=smc_metrics,
        )

        if not should_continue:
            print("Training run halted by monitor.")
            break

    trainer.close() 
