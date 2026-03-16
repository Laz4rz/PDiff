"""Module entrypoint and Hydra-based CLI for discrete diffusion.

Usage:
  python -m discrete_diffusion [Hydra overrides]
"""

import os
import math
import json
from pathlib import Path

import hydra
import lightning as L
import omegaconf
import torch
import fsspec
import rich.syntax
import rich.tree

from .train import train as train_function
from . import utils


CONFIG_PATH = (Path(__file__).resolve().parents[2] / "configs").as_posix()


def _register_resolver(name, resolver):
    if omegaconf.OmegaConf.has_resolver(name):
        return
    omegaconf.OmegaConf.register_new_resolver(name, resolver)


def _mul_resolver(*args):
    import functools, operator

    return (
        functools.reduce(operator.mul, args)
        if args
        else ValueError("`mul` resolver requires at least one argument.")
    )


def _lr_from_log10_resolver(log10_lr, default_lr):
    if log10_lr is None:
        return float(default_lr)
    if isinstance(log10_lr, str) and log10_lr.strip().lower() in {"", "none", "null"}:
        return float(default_lr)
    return 10 ** float(log10_lr)


# Register OmegaConf resolvers for Hydra configs
_register_resolver("cwd", os.getcwd)
_register_resolver("device_count", torch.cuda.device_count)
_register_resolver("div_up", lambda x, y: (x + y - 1) // y)
_register_resolver("mul", _mul_resolver)
_register_resolver("sub", lambda x, y: x - y)
_register_resolver("pow10", lambda x: 10 ** float(x))
_register_resolver("exp", lambda x: math.exp(float(x)))
_register_resolver("lr_from_log10", _lr_from_log10_resolver)


def _runtime_repro_context(config: omegaconf.DictConfig) -> dict:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_TMPDIR",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "PDIFF_DISPATCH_SIGNATURE",
        "PDIFF_RUN_INDEX",
    ]
    env_values = {}
    for key in env_keys:
        value = os.environ.get(key)
        if value not in (None, ""):
            env_values[key] = value

    device_name = None
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except RuntimeError:
            device_name = "<unavailable>"

    return {
        "seed": omegaconf.OmegaConf.select(config, "seed", default=None),
        "lr_log10": omegaconf.OmegaConf.select(config, "lr_log10", default=None),
        "optim_lr": omegaconf.OmegaConf.select(config, "optim.lr", default=None),
        "trainer_precision": omegaconf.OmegaConf.select(
            config, "trainer.precision", default=None
        ),
        "trainer_deterministic": omegaconf.OmegaConf.select(
            config, "trainer.deterministic", default=None
        ),
        "force_deterministic_algorithms": omegaconf.OmegaConf.select(
            config, "training.force_deterministic_algorithms", default=False
        ),
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
            if torch.cuda.is_available()
            else None
        ),
        "cudnn_allow_tf32": (
            torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name_0": device_name,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "env": env_values,
    }


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(f"{config.checkpointing.save_dir}/config_tree.txt", "w") as fp:
            rich.print(tree, file=fp)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(config):
    logger = utils.get_logger(__name__)
    force_deterministic_algos = bool(
        omegaconf.OmegaConf.select(
            config, "training.force_deterministic_algorithms", default=False
        )
    )
    if force_deterministic_algos:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(force_deterministic_algos)
    torch.backends.cudnn.deterministic = force_deterministic_algos
    if force_deterministic_algos:
        torch.backends.cudnn.benchmark = False

    L.seed_everything(config.seed, workers=True)
    logger.info(
        "Runtime reproducibility context: %s",
        json.dumps(_runtime_repro_context(config), sort_keys=True),
    )
    should_print_config = omegaconf.OmegaConf.select(
        config, "logging.print_config", default=True
    )
    if should_print_config:
        _print_config(config)

    logger.info("Starting training...")
    train_function(config)
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
