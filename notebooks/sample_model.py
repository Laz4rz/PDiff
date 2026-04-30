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
    try:
        tokenized = tokenizer(
            prompts, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
    except (NotImplementedError, TypeError, AttributeError):
        # BrevoDummyTokenizer exposes decode-only behavior; treat prompts as pretokenized
        # whitespace-delimited ids in that case.
        tokenized = []
        for prompt in prompts:
            stripped = prompt.strip()
            if not stripped:
                tokenized.append([])
                continue
            try:
                tokenized.append([int(tok) for tok in stripped.split()])
            except ValueError as exc:
                raise TypeError(
                    "Prefix prompts must be whitespace-delimited token ids when tokenizer "
                    "does not support tokenization."
                ) from exc
    bos_id = tokenizer.bos_token_id
    prefix_token_ids = []
    for ids in tokenized:
        ids = list(ids)
        if bos_id is not None and (len(ids) == 0 or ids[0] != bos_id):
            ids = [bos_id] + ids
        prefix_token_ids.append(ids)
    return prompts, prefix_token_ids


def _build_prefix_constraints(
    prefix_token_ids: Sequence[Sequence[int]],
    *,
    num_samples: int,
    seq_len: int,
    device,
    dtype,
):
    fixed_tokens = torch.zeros((num_samples, seq_len), device=device, dtype=dtype)
    fixed_mask = torch.zeros((num_samples, seq_len), device=device, dtype=torch.bool)
    for row_idx, ids in enumerate(prefix_token_ids):
        if row_idx >= num_samples:
            break
        prefix_len = min(len(ids), seq_len)
        if prefix_len == 0:
            continue
        fixed_tokens[row_idx, :prefix_len] = torch.as_tensor(
            ids[:prefix_len], device=device, dtype=dtype
        )
        fixed_mask[row_idx, :prefix_len] = True
    return fixed_tokens, fixed_mask


def _print_prefix_completions(
    samples,
    tokenizer,
    prefix_prompts,
    prefix_token_ids,
    expected_completions: Sequence[str] | None = None,
):
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
        if expected_completions is not None:
            print(f"expected: {expected_completions[i]}")
        print()


def print_run_metadata(ckpt_path: Path, ckpt_cfg) -> None:
    print("Checkpoint:", ckpt_path)
    print("Algo:", ckpt_cfg.algo.name)
    print("Loss Type:", ckpt_cfg.algo.loss_type)
    print("P_u=", ckpt_cfg.algo.p_uniform)


def _resolve_sampling_params(model, num_steps: int | None, eps: float | None):
    if num_steps is None:
        num_steps = int(model.config.sampling.steps)
    if eps is None:
        eps = 1e-5
    return int(num_steps), float(eps)


def _is_gidd_sampler(sampler) -> bool:
    return sampler is not None and sampler.__class__.__name__ == "GIDDSampler"


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
    prefix_prompt_texts = None
    prefix_token_ids = None
    fixed_tokens = None
    fixed_mask = None
    sampler = model._create_sampler()
    is_gidd = _is_gidd_sampler(sampler)

    if prefix_prompts is not None:
        if not is_gidd:
            raise NotImplementedError(
                "Prefix-conditioned sampling in this notebook currently supports GIDDSampler only."
            )
        prefix_prompt_texts, prefix_token_ids = _tokenize_prefix_prompts(
            tokenizer, prefix_prompts, num_samples
        )
        fixed_tokens, fixed_mask = _build_prefix_constraints(
            prefix_token_ids,
            num_samples=num_samples,
            seq_len=model.num_tokens,
            device=model.device,
            dtype=torch.long,
        )

    if not show_steps:
        if is_gidd and fixed_tokens is not None:
            samples = model.generate_samples(
                num_samples=num_samples,
                num_steps=num_steps,
                eps=eps,
                fixed_tokens=fixed_tokens,
                fixed_mask=fixed_mask,
            )
        else:
            samples = model.generate_samples(
                num_samples=num_samples,
                num_steps=num_steps,
                eps=eps,
            )
        return samples, prefix_prompt_texts, prefix_token_ids

    resolved_steps, resolved_eps = _resolve_sampling_params(model, num_steps, eps)

    if is_gidd:
        if fixed_tokens is None:
            trajectory = model.generate_samples(
                num_samples=num_samples,
                num_steps=resolved_steps,
                eps=resolved_eps,
                record_steps=True,
            )
        else:
            trajectory = model.generate_samples(
                num_samples=num_samples,
                num_steps=resolved_steps,
                eps=resolved_eps,
                record_steps=True,
                fixed_tokens=fixed_tokens,
                fixed_mask=fixed_mask,
            )
        timesteps = torch.linspace(1 - resolved_eps, resolved_eps, resolved_steps + 1)
        step_every = max(1, int(step_every))
        for step_idx, z_t in enumerate(trajectory):
            if (
                step_idx == 0
                or step_idx == resolved_steps
                or step_idx % step_every == 0
            ):
                _decode_step(
                    step_idx,
                    resolved_steps,
                    timesteps[step_idx].item(),
                    z_t,
                    tokenizer,
                    max_samples=max_samples,
                    skip_special_tokens=skip_special_tokens,
                )
        samples = trajectory[-1].to(model.device)
    else:
        samples = model.generate_samples(
            num_samples=num_samples,
            num_steps=resolved_steps,
            eps=resolved_eps,
        )
        _decode_step(
            resolved_steps,
            resolved_steps,
            resolved_eps,
            samples,
            tokenizer,
            max_samples=max_samples,
            skip_special_tokens=skip_special_tokens,
        )
    return samples, prefix_prompt_texts, prefix_token_ids


def print_samples(
    samples,
    tokenizer,
    *,
    prefix_prompt_texts: Sequence[str] | None = None,
    prefix_token_ids: Sequence[Sequence[int]] | None = None,
    expected_completions: Sequence[str] | None = None,
    skip_special_tokens: bool = True,
) -> None:
    if prefix_token_ids is not None and prefix_prompt_texts is not None:
        if expected_completions is not None:
            expected_completions = _expand_prefix_prompts(
                expected_completions, len(prefix_prompt_texts)
            )
        _print_prefix_completions(
            samples,
            tokenizer,
            prefix_prompt_texts,
            prefix_token_ids,
            expected_completions=expected_completions,
        )
        return

    texts = tokenizer.batch_decode(
        samples.detach().cpu(), skip_special_tokens=skip_special_tokens
    )
    for i, text in enumerate(texts):
        print(f"--- sample {i} ---")
        print(text)
        print()
