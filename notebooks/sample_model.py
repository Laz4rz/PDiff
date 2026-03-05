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


NUM_SAMPLES = 24
NUM_STEPS = None  # use model default
# RUN_DIR = ROOT / "outputs/roneneldan/TinyStories/2026.02.05/194344"
# RUN_DIR = ROOT / "outputs/roneneldan/TinyStories/2026.02.05/194409"
# RUN_DIR = ROOT / "outputs/prefix/2026.03.04/210842"
RUN_DIR = ROOT / "outputs/prefix/2026.03.05/144345"
# Set prompts to enable prefix-conditioned sampling (None keeps unconditional sampling).
PREFIX_PROMPTS: list[str] | None = None
PREFIX_PROMPTS = [
    "The capital of France is:",
    "The capital of Germany is:",
    "The capital of Italy is:",
    "The capital of Spain is:",
    "The capital of Japan is:",
    "The capital of Canada is:",
    "The capital of Australia is:",
    "2 + 2 =",
    "5 * 6 =",
    "9 - 4 =",
    "The opposite of hot is:",
    "The opposite of up is:",
    "The color of the sky on a clear day is:",
    "The first day of the week in the ISO standard is:",
    "The largest planet in our solar system is:",
    "Water freezes at 0 degrees:",
    "The chemical symbol for gold is:",
    "The language mostly spoken in Brazil is:",
    "The author of '1984' is:",
    "The square root of 81 is:",
    "The next letter after C is:",
    "The past tense of 'go' is:",
    "A baby cat is called a:",
    "A shape with three sides is a:",
]
# PREFIX_PROMPTS = ["The capital of France is:"]
SHOW_STEPS = True
STEP_EVERY = 64
STEP_MAX_SAMPLES = 24
SKIP_SPECIAL_TOKENS = False
EPS = None  # set to a float to override (e.g., 1e-5)
WRAP_WIDTH = 100


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


model, tokenizer, ckpt_path, ckpt_cfg, hydra_cfg = load_model_from_run(RUN_DIR)
print("Checkpoint:", ckpt_path)
print("Algo:", ckpt_cfg.algo.name)
print("Loss Type:", ckpt_cfg.algo.loss_type)
print("P_u=", ckpt_cfg.algo.p_uniform)

if SHOW_STEPS:
    samples, prefix_prompt_texts, prefix_token_ids = generate_with_steps(
        model,
        tokenizer,
        num_samples=NUM_SAMPLES,
        num_steps=NUM_STEPS,
        eps=EPS,
        step_every=STEP_EVERY,
        max_samples=STEP_MAX_SAMPLES,
        skip_special_tokens=SKIP_SPECIAL_TOKENS,
        show_steps=True,
        prefix_prompts=PREFIX_PROMPTS,
    )
else:
    if PREFIX_PROMPTS is None:
        samples = model.generate_samples(
            num_samples=NUM_SAMPLES, num_steps=NUM_STEPS, eps=EPS
        )
        prefix_prompt_texts, prefix_token_ids = None, None
    else:
        samples, prefix_prompt_texts, prefix_token_ids = generate_with_steps(
            model,
            tokenizer,
            num_samples=NUM_SAMPLES,
            num_steps=NUM_STEPS,
            eps=EPS,
            step_every=STEP_EVERY,
            max_samples=STEP_MAX_SAMPLES,
            skip_special_tokens=SKIP_SPECIAL_TOKENS,
            show_steps=False,
            prefix_prompts=PREFIX_PROMPTS,
        )

if prefix_token_ids is not None:
    _print_prefix_completions(
        samples, tokenizer, prefix_prompt_texts, prefix_token_ids
    )
else:
    texts = tokenizer.batch_decode(samples.detach().cpu(), skip_special_tokens=True)
    for i, text in enumerate(texts):
        print(f"--- sample {i} ---")
        print(text)
        print()
