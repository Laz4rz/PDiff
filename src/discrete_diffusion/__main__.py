"""Module entrypoint and Hydra-based CLI for discrete diffusion.

Usage:
  python -m discrete_diffusion [Hydra overrides]
"""

import os
import math
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
    should_print_config = omegaconf.OmegaConf.select(
        config, "logging.print_config", default=True
    )
    if should_print_config:
        _print_config(config)

    logger = utils.get_logger(__name__)
    logger.info("Starting training...")
    train_function(config)
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
