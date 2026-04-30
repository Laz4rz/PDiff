from pathlib import Path
from itertools import islice
from typing import Literal

import hydra
import torch

from sample_model import ROOT, print_run_metadata, print_samples, run_sampling

from discrete_diffusion.data import get_tokenizer
from discrete_diffusion.data.datasets import iter_brevo_split_records

CKPT_TYPE: Literal["best", "last"] = "best"
RUN_DIR = ROOT / "outputs/brevo/2026.04.15/160409"
NUM_SAMPLES = 8
NUM_STEPS = 128
SHOW_STEPS = False
STEP_EVERY = 16
STEP_MAX_SAMPLES = 8
SKIP_SPECIAL_TOKENS = False
EPS = None  # set float to override, e.g. 1e-5


def _resolve_ckpt_path(run_dir: Path, ckpt_type: Literal["best", "last"]) -> Path:
    candidates = [
        run_dir / "checkpoints" / f"{ckpt_type}.ckpt",
        run_dir / "dummy_checkpoints" / "checkpoints" / f"{ckpt_type}.ckpt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {ckpt_type}.ckpt under {run_dir}/checkpoints or "
        f"{run_dir}/dummy_checkpoints/checkpoints."
    )


def load_brevo_model(run_dir: Path, ckpt_type: Literal["best", "last"] = "best"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_ckpt_path(Path(run_dir), ckpt_type)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["hyper_parameters"]["config"]

    tokenizer = get_tokenizer(ckpt_cfg)
    algo_cls = hydra.utils.get_class(ckpt_cfg.algo._target_)
    model = algo_cls.load_from_checkpoint(
        ckpt_path,
        config=ckpt_cfg,
        tokenizer=tokenizer,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, ckpt_path, ckpt_cfg


def _resolve_brevo_dataset_config(ckpt_cfg) -> dict:
    data_cfg = ckpt_cfg.get("data", {}) if isinstance(ckpt_cfg, dict) else ckpt_cfg.data
    raw_cfg = (
        data_cfg.get("dataset_config", {})
        if isinstance(data_cfg, dict)
        else getattr(data_cfg, "dataset_config", {})
    )
    if raw_cfg is None:
        return {}
    try:
        cfg = dict(raw_cfg)
    except TypeError:
        return {}
    return {k: v for k, v in cfg.items() if v not in (None, "")}


if not torch.cuda.is_available():
    raise RuntimeError(
        "Brevo sampling requires CUDA in this setup (flash-attn rotary kernels)."
    )

model, tokenizer, ckpt_path, ckpt_cfg = load_brevo_model(RUN_DIR, CKPT_TYPE)
print_run_metadata(ckpt_path, ckpt_cfg)

brevo_dataset_config = _resolve_brevo_dataset_config(ckpt_cfg)
validation_records = list(
    islice(
        iter_brevo_split_records(
            dataset_config=brevo_dataset_config,
            split="validation",
        ),
        NUM_SAMPLES,
    )
)
if not validation_records:
    raise RuntimeError("Validation split produced no records.")
prefixes = [record["prefixes"] for record in validation_records]
completions = [record["completions"] for record in validation_records]

samples, prefix_prompt_texts, prefix_token_ids = run_sampling(
    model,
    tokenizer,
    num_samples=len(prefixes),
    num_steps=NUM_STEPS,
    eps=EPS,
    step_every=STEP_EVERY,
    max_samples=STEP_MAX_SAMPLES,
    skip_special_tokens=SKIP_SPECIAL_TOKENS,
    show_steps=SHOW_STEPS,
    prefix_prompts=prefixes,
)

print_samples(
    samples,
    tokenizer,
    prefix_prompt_texts=prefix_prompt_texts,
    prefix_token_ids=prefix_token_ids,
    expected_completions=completions,
    skip_special_tokens=SKIP_SPECIAL_TOKENS,
)
