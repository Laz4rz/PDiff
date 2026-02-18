"""Public training API for discrete diffusion models."""

import json
import os
from pathlib import Path

import hydra
import lightning as L
import omegaconf
import torch

from .data import get_dataloaders, get_tokenizer
from . import utils


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


def train(config):
    """Main training API.

    Args:
      config: Hydra DictConfig or config object with training parameters.

    Returns:
      None. Model checkpoints are saved according to config.checkpointing.
    """
    # Set matmul precision to 'high' (TF32) to match FlexMDM
    torch.set_float32_matmul_precision("high")

    logger = utils.get_logger(__name__)
    logger.info("Starting Training.")

    tokenizer = get_tokenizer(config)
    algo_cls = hydra.utils.get_class(config.algo._target_)

    # Ensure dataset processing happens on rank 0 first
    fabric = L.Fabric(
        num_nodes=config.trainer.num_nodes,
        devices=config.trainer.devices,
        accelerator="cuda",
    )
    fabric.launch()
    with fabric.rank_zero_first():
        train_ds, valid_ds = get_dataloaders(config, tokenizer)
    fabric.barrier()
    del fabric

    # WandB logger
    wandb_logger = (
        L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb
        )
        if config.get("wandb", None) is not None
        else None
    )

    # Resume checkpoint path
    ckpt_path = (
        config.checkpointing.resume_ckpt_path
        if (
            config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )
        else None
    )

    # Lightning callbacks
    callbacks = (
        [hydra.utils.instantiate(cb) for _, cb in config.callbacks.items()]
        if "callbacks" in config
        else []
    )

    if config.training.finetune_path != "":
        assert utils.fsspec_exists(config.training.finetune_path)
        model = algo_cls.load_from_checkpoint(
            config.training.finetune_path, tokenizer=tokenizer, config=config
        )
    else:
        model = algo_cls(config, tokenizer=tokenizer)

    # Torch compile if enabled
    if omegaconf.OmegaConf.select(config, "training.torch_compile", default=False):
        logger.info("Compiling LightningModule with torch.compile.")
        model = torch.compile(model)

    if config.training.get("fault_tolerant", False):
        os.environ.setdefault("PL_FAULT_TOLERANT_TRAINING", "1")

    trainer = L.Trainer(
        **config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger,
    )
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)

    run_train_end_validation = omegaconf.OmegaConf.select(
        config, "eval.run_validation_at_train_end", default=True
    )
    if run_train_end_validation:
        validation_ckpt_path = omegaconf.OmegaConf.select(
            config, "eval.train_end_validation_ckpt_path", default="last"
        )
        metrics_path = omegaconf.OmegaConf.select(
            config,
            "eval.train_end_validation_metrics_path",
            default=f"{os.getcwd()}/train_end_validation_metrics.json",
        )

        validation_metrics = trainer.validate(
            model=model, dataloaders=valid_ds, ckpt_path=validation_ckpt_path
        )

        if trainer.is_global_zero:
            payload = {
                "global_step": int(trainer.global_step),
                "current_epoch": int(trainer.current_epoch),
                "ckpt_path": validation_ckpt_path,
                "results": _to_jsonable(validation_metrics),
            }
            metrics_file = Path(metrics_path)
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_file, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2, sort_keys=True)
