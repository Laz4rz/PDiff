import sys
import textwrap
from pathlib import Path
from typing import Literal, Sequence

import hydra
import torch
from omegaconf import OmegaConf


def _repo_root() -> Path:
    try:
        here = Path(__file__).resolve()
        return here.parents[1]
    except NameError:
        # Fallback for interactive contexts
        return Path(".").resolve()


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from discrete_diffusion.data import get_tokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

WRAP_WIDTH = 100


def load_model_from_run(run_dir: Path, ckpt_type: Literal["best", "last"] = "best"):
    ckpt_path = (
        Path(run_dir) / "dummy_checkpoints" / "checkpoints" / f"{ckpt_type}.ckpt"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["hyper_parameters"]["config"]
    hydra_cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")

    tokenizer = get_tokenizer(ckpt_cfg)
    algo_target = ckpt_cfg.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)
    model = algo_cls.load_from_checkpoint(
        ckpt_path,
        config=ckpt_cfg,
        tokenizer=tokenizer,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, ckpt_path, ckpt_cfg, hydra_cfg


def _decode_step(
    step_idx, num_steps, t_val, z_t, tokenizer, *, max_samples, skip_special_tokens
):
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is not None:
        mask_rate = (z_t == mask_id).float().mean().item() * 100
        header = (
            f"Step {step_idx:>4}/{num_steps:<4}  t={t_val:.6f}  mask={mask_rate:>5.1f}%"
        )
    else:
        header = f"Step {step_idx:>4}/{num_steps:<4}  t={t_val:.6f}"
    print("=" * max(60, len(header)))
    print(header)
    print("-" * max(60, len(header)))
    texts = tokenizer.batch_decode(
        z_t.detach().cpu(),
        skip_special_tokens=skip_special_tokens,
    )
    for i, text in enumerate(texts[:max_samples]):
        prefix = f"[{i:02d}] "
        wrapped = textwrap.fill(
            text,
            width=WRAP_WIDTH,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
        )
        print(wrapped)
    print()


def _expand_prefix_prompts(prefix_prompts: Sequence[str], num_samples: int) -> list[str]:
    if len(prefix_prompts) == 0:
        raise ValueError("PREFIX_PROMPTS must contain at least one prompt.")
    if len(prefix_prompts) == num_samples:
        return list(prefix_prompts)
    if len(prefix_prompts) == 1:
        return [prefix_prompts[0]] * num_samples
    return [prefix_prompts[i % len(prefix_prompts)] for i in range(num_samples)]


def _tokenize_prefix_prompts(
    tokenizer,
    prefix_prompts: Sequence[str],
    num_samples: int,
) -> tuple[list[str], list[list[int]]]:
    prompts = _expand_prefix_prompts(prefix_prompts, num_samples)
    tokenized = tokenizer(
        prompts, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    bos_id = tokenizer.bos_token_id
    prefix_token_ids = []
    for ids in tokenized:
        ids = list(ids)
        if bos_id is not None and (len(ids) == 0 or ids[0] != bos_id):
            ids = [bos_id] + ids
        prefix_token_ids.append(ids)
    return prompts, prefix_token_ids


def _apply_prefix_constraints(z_t: torch.Tensor, prefix_token_ids: Sequence[Sequence[int]]):
    seq_len = z_t.shape[1]
    for row_idx, ids in enumerate(prefix_token_ids):
        prefix_len = min(len(ids), seq_len)
        if prefix_len == 0:
            continue
        z_t[row_idx, :prefix_len] = torch.as_tensor(
            ids[:prefix_len], device=z_t.device, dtype=z_t.dtype
        )
    return z_t


def _print_prefix_completions(samples, tokenizer, prefix_prompts, prefix_token_ids):
    eos_id = tokenizer.eos_token_id
    samples_cpu = samples.detach().cpu()
    for i, (prompt, prefix_ids) in enumerate(zip(prefix_prompts, prefix_token_ids)):
        seq = samples_cpu[i].tolist()
        suffix = seq[min(len(prefix_ids), len(seq)) :]
        if eos_id is not None and eos_id in suffix:
            suffix = suffix[: suffix.index(eos_id)]
        completion = tokenizer.decode(suffix, skip_special_tokens=True).strip()
        print(f"--- sample {i} ---")
        print(f"prefix: {prompt}")
        print(f"completion: {completion}")
        print()


@torch.no_grad()
def generate_with_steps(
    model,
    tokenizer,
    *,
    num_samples,
    num_steps,
    eps,
    step_every,
    max_samples,
    skip_special_tokens,
    show_steps,
    prefix_prompts,
):
    sampler = model._create_sampler()
    prefix_prompt_texts = None
    prefix_token_ids = None
    if sampler is None or sampler.__class__.__name__ != "GIDDSampler":
        if prefix_prompts is not None:
            raise NotImplementedError(
                "Prefix-conditioned sampling in this notebook currently supports GIDDSampler only."
            )
        samples = model.generate_samples(
            num_samples=num_samples, num_steps=num_steps, eps=eps
        )
        if show_steps:
            texts = tokenizer.batch_decode(samples.detach().cpu(), skip_special_tokens=True)
            for i, text in enumerate(texts):
                print(f"--- sample {i} ---")
                print(text)
                print()
        return samples, prefix_prompt_texts, prefix_token_ids

    if num_steps is None:
        num_steps = model.config.sampling.steps
    if eps is None:
        eps = 1e-5
    inject_bos = getattr(model.config.sampling, "inject_bos", True)

    z_t = model.hybrid_noise.sample_prior((num_samples, model.num_tokens)).to(
        model.device
    )
    if inject_bos:
        z_t[:, 0] = model.tokenizer.bos_token_id
    if prefix_prompts is not None:
        prefix_prompt_texts, prefix_token_ids = _tokenize_prefix_prompts(
            tokenizer, prefix_prompts, num_samples
        )
        z_t = _apply_prefix_constraints(z_t, prefix_token_ids)

    timesteps = torch.linspace(1 - eps, eps, num_steps + 1, device=model.device)
    if show_steps:
        _decode_step(
            0,
            num_steps,
            timesteps[0].item(),
            z_t,
            tokenizer,
            max_samples=max_samples,
            skip_special_tokens=skip_special_tokens,
        )

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(num_samples, device=model.device)
        s = timesteps[i + 1] * torch.ones(num_samples, device=model.device)
        z_t = sampler.compute_posterior(model, z_t, t, s)
        if prefix_token_ids is not None:
            z_t = _apply_prefix_constraints(z_t, prefix_token_ids)
        if show_steps and ((i + 1) % step_every == 0 or i == num_steps - 1):
            _decode_step(
                i + 1,
                num_steps,
                timesteps[i + 1].item(),
                z_t,
                tokenizer,
                max_samples=max_samples,
                skip_special_tokens=skip_special_tokens,
            )
    return z_t, prefix_prompt_texts, prefix_token_ids


def print_run_metadata(ckpt_path: Path, ckpt_cfg) -> None:
    print("Checkpoint:", ckpt_path)
    print("Algo:", ckpt_cfg.algo.name)
    print("Loss Type:", ckpt_cfg.algo.loss_type)
    print("P_u=", ckpt_cfg.algo.p_uniform)


@torch.no_grad()
def run_sampling(
    model,
    tokenizer,
    *,
    num_samples: int,
    num_steps: int | None = None,
    eps: float | None = None,
    step_every: int = 64,
    max_samples: int = 24,
    skip_special_tokens: bool = False,
    show_steps: bool = True,
    prefix_prompts: Sequence[str] | None = None,
):
    if show_steps:
        return generate_with_steps(
            model,
            tokenizer,
            num_samples=num_samples,
            num_steps=num_steps,
            eps=eps,
            step_every=step_every,
            max_samples=max_samples,
            skip_special_tokens=skip_special_tokens,
            show_steps=True,
            prefix_prompts=prefix_prompts,
        )

    if prefix_prompts is None:
        samples = model.generate_samples(num_samples=num_samples, num_steps=num_steps, eps=eps)
        return samples, None, None

    return generate_with_steps(
        model,
        tokenizer,
        num_samples=num_samples,
        num_steps=num_steps,
        eps=eps,
        step_every=step_every,
        max_samples=max_samples,
        skip_special_tokens=skip_special_tokens,
        show_steps=False,
        prefix_prompts=prefix_prompts,
    )


def print_samples(
    samples,
    tokenizer,
    *,
    prefix_prompt_texts: Sequence[str] | None = None,
    prefix_token_ids: Sequence[Sequence[int]] | None = None,
    skip_special_tokens: bool = True,
) -> None:
    if prefix_token_ids is not None and prefix_prompt_texts is not None:
        _print_prefix_completions(
            samples, tokenizer, prefix_prompt_texts, prefix_token_ids
        )
        return

    texts = tokenizer.batch_decode(
        samples.detach().cpu(), skip_special_tokens=skip_special_tokens
    )
    for i, text in enumerate(texts):
        print(f"--- sample {i} ---")
        print(text)
        print()
