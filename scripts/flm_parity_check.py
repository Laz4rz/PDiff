#!/usr/bin/env python
"""Compare the PDiff FLM path against the reference FLM repository.

The check intentionally runs each implementation in an isolated subprocess so
imports and Hydra resolvers do not leak between repos.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _probe_code(
    kind: str, repo: Path, cache_dir: Path, steps: int, use_data: bool
) -> str:
    if kind == "source":
        imports = """
import main  # registers source OmegaConf resolvers
import dataloader
import algo
"""
        config_name = "config_wikitext2v1_flm_low_compute"
        tokenizer_expr = "dataloader.get_tokenizer(cfg)"
        loader_setup = """
train_loader = valid_loader = None
if USE_DATA:
    train_loader, valid_loader = dataloader.get_dataloaders(cfg, tokenizer)
"""
        model_expr = "algo.FLM(cfg, tokenizer=tokenizer)"
    elif kind == "pdiff":
        imports = """
import discrete_diffusion.__main__  # registers PDiff OmegaConf resolvers
from discrete_diffusion.data import get_tokenizer, get_dataloaders
from discrete_diffusion.data import loaders as loader_mod
from discrete_diffusion.algorithms.flm import FLM
loader_mod.LOAD_FROM_CACHE = True
"""
        config_name = "config_wikitext2v1_flm_low_compute"
        tokenizer_expr = "get_tokenizer(cfg)"
        loader_setup = """
train_loader = valid_loader = None
if USE_DATA:
    train_loader, valid_loader = get_dataloaders(cfg, tokenizer)
"""
        model_expr = "FLM(cfg, tokenizer=tokenizer)"
    else:
        raise ValueError(kind)

    config_dir = repo / "configs"
    if kind == "pdiff":
        path_setup = f"sys.path.insert(0, {str(repo / 'src')!r})"
    else:
        path_setup = f"sys.path.insert(0, {str(repo)!r})"
    imports = textwrap.dedent(imports).strip()
    loader_setup = textwrap.dedent(loader_setup).strip()

    return f"""
import hashlib
import json
import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

{path_setup}
{imports}

CONFIG_DIR = {str(config_dir)!r}
CACHE_DIR = {str(cache_dir)!r}
STEPS = {int(steps)}
USE_DATA = {bool(use_data)!r}

def hash_tensor(tensor):
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()

with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg = compose(
        config_name={config_name!r},
        overrides=[
            f"data.cache_dir={{CACHE_DIR}}",
            "wandb=null",
            "++trainer.enable_progress_bar=false",
            "++trainer.enable_model_summary=false",
        ],
    )
L.seed_everything(cfg.seed, workers=False)

tokenizer = {tokenizer_expr}
{loader_setup}

model = {model_expr}
model.train()

if USE_DATA:
    batch = next(iter(train_loader))
else:
    torch.manual_seed(cfg.seed)
    batch = {{
        "input_ids": torch.randint(
            0, len(tokenizer), (cfg.loader.batch_size, cfg.model.length)
        ),
        "attention_mask": torch.ones(
            cfg.loader.batch_size, cfg.model.length, dtype=torch.long
        ),
    }}

input_ids = batch["input_ids"].long()
attention_mask = batch["attention_mask"].long()

state_shapes = {{
    key: list(value.shape)
    for key, value in model.backbone.state_dict().items()
}}
params = [p for p in model._get_parameters() if p.requires_grad]
param_count = sum(p.numel() for p in params)

first_loss = model._loss(input_ids, attention_mask, train_mode=True).loss
step_losses = []
if STEPS > 0:
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    loader_iter = iter(train_loader) if USE_DATA else None
    for _ in range(STEPS):
        if USE_DATA:
            step_batch = next(loader_iter)
            step_input_ids = step_batch["input_ids"].long()
            step_attention_mask = step_batch["attention_mask"].long()
        else:
            step_input_ids = input_ids
            step_attention_mask = attention_mask
        optimizer.zero_grad(set_to_none=True)
        loss = model._loss(step_input_ids, step_attention_mask, train_mode=True).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.trainer.gradient_clip_val)
        optimizer.step()
        scheduler.step()
        step_losses.append(float(loss.detach().cpu()))

print(json.dumps({{
    "kind": {kind!r},
    "tokenizer_length": len(tokenizer),
    "batch_hash": hash_tensor(input_ids) if USE_DATA else None,
    "attention_hash": hash_tensor(attention_mask) if USE_DATA else None,
    "param_count": int(param_count),
    "state_shapes": state_shapes,
    "first_loss": float(first_loss.detach().cpu()),
    "step_losses": step_losses,
    "config": {{
        "model_length": int(cfg.model.length),
        "n_blocks": int(cfg.model.n_blocks),
        "n_heads": int(cfg.model.n_heads),
        "lr": float(cfg.optim.lr),
        "warmup": int(cfg.lr_scheduler.num_warmup_steps),
        "batch_size": int(cfg.loader.batch_size),
        "add_mask_token": bool(cfg.data.get("add_mask_token", False)),
    }},
}}, sort_keys=True))
"""


def _run_probe(
    kind: str,
    repo: Path,
    cache_dir: Path,
    steps: int,
    use_data: bool,
    python_executable: Path,
):
    code = _probe_code(kind, repo, cache_dir, steps, use_data)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(
        [str(python_executable), "-c", code],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{kind} probe failed with code {result.returncode}\\n"
            f"STDOUT:\\n{result.stdout}\\nSTDERR:\\n{result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _normalize_python_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _assert_close(name: str, left: float, right: float, rtol: float, atol: float):
    if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{name} mismatch: source={left}, pdiff={right}")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_pdiff = Path(__file__).resolve().parents[1]
    parser.add_argument("--pdiff-repo", type=Path, default=default_pdiff)
    parser.add_argument("--flm-repo", type=Path, default=default_pdiff.parent / "flm")
    parser.add_argument("--pdiff-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--flm-python", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    pdiff_repo = args.pdiff_repo.resolve()
    flm_repo = args.flm_repo.resolve()
    pdiff_python = _normalize_python_path(args.pdiff_python)
    flm_python = (
        _normalize_python_path(args.flm_python)
        if args.flm_python is not None
        else flm_repo / ".venv" / "bin" / "python"
    )
    if not flm_python.exists():
        flm_python = Path(sys.executable)
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else flm_repo / ".cache" / "wikitext2-v1"
    )
    use_data = not args.no_data

    source = _run_probe("source", flm_repo, cache_dir, args.steps, use_data, flm_python)
    pdiff = _run_probe(
        "pdiff", pdiff_repo, cache_dir, args.steps, use_data, pdiff_python
    )

    for key in ("tokenizer_length", "param_count", "config"):
        if source[key] != pdiff[key]:
            raise AssertionError(
                f"{key} mismatch: source={source[key]}, pdiff={pdiff[key]}"
            )
    if source["state_shapes"] != pdiff["state_shapes"]:
        raise AssertionError("model state shape dictionaries differ")
    if use_data:
        for key in ("batch_hash", "attention_hash"):
            if source[key] != pdiff[key]:
                raise AssertionError(
                    f"{key} mismatch: source={source[key]}, pdiff={pdiff[key]}"
                )

    _assert_close(
        "first_loss", source["first_loss"], pdiff["first_loss"], args.rtol, args.atol
    )
    if len(source["step_losses"]) != len(pdiff["step_losses"]):
        raise AssertionError("step loss lengths differ")
    for idx, (src_loss, pdiff_loss) in enumerate(
        zip(source["step_losses"], pdiff["step_losses"])
    ):
        _assert_close(f"step_losses[{idx}]", src_loss, pdiff_loss, args.rtol, args.atol)

    print(json.dumps({"source": source, "pdiff": pdiff}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
