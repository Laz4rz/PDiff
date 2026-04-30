from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path
from typing import Literal

import hydra
import torch
from omegaconf import OmegaConf


def _repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path(".").resolve()


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from discrete_diffusion.data import get_tokenizer  # noqa: E402
from discrete_diffusion.data.datasets import iter_brevo_split_records  # noqa: E402
from discrete_diffusion.data.generators.brevo import TopoSortDepthStats  # noqa: E402
from discrete_diffusion.sampling.gidd import (  # noqa: E402
    GIDDSampler,
    GIDDLeftToRightSampler,
    GIDDRightToLeftSampler,
    GIDDSingleTokenSampler,
)


GRAPH_NS = (5, 10, 20)
CKPT_TYPE: Literal["best", "last"] = "best"
NUM_SAMPLES = 3
NUM_STEPS = 128
EPS = None
STEP_EVERY = 1
PRINT_TEXT = False
OUT_DIR = ROOT / "notebooks" / "sampling_traces" / "brevo_gidd_sampler_steps"
SKIP_MISSING_CHECKPOINTS = True

# Override these if you want exact runs. The auto-discovery below prefers local
# checkpointed tiny GIDD runs for each graph size.
RUN_DIRS: dict[int, Path] = {}

SAMPLERS = (
    ("ancestral", GIDDSampler),
    ("left_to_right", GIDDLeftToRightSampler),
    ("right_to_left", GIDDRightToLeftSampler),
    ("single_token", GIDDSingleTokenSampler),
)


def _resolve_ckpt_path(run_dir: Path, ckpt_type: Literal["best", "last"]) -> Path:
    candidates = [
        run_dir / "checkpoints" / f"{ckpt_type}.ckpt",
        run_dir / "dummy_checkpoints" / "checkpoints" / f"{ckpt_type}.ckpt",
    ]
    for path in candidates:
        if path.exists():
            return path

    final_metrics = run_dir / "final_metrics.json"
    extra = ""
    if final_metrics.exists():
        extra = (
            f" {final_metrics} exists, but its recorded checkpoint file is not present "
            "in the local tree."
        )
    raise FileNotFoundError(f"No {ckpt_type}.ckpt found under {run_dir}.{extra}")


def _cfg_value(cfg, path: str, default=None):
    return OmegaConf.select(cfg, path, default=default)


def _is_tiny_gidd_brevo_run(run_dir: Path, graph_n: int) -> bool:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return False
    cfg = OmegaConf.load(cfg_path)
    target = str(_cfg_value(cfg, "algo._target_", ""))
    return (
        target.endswith(".gidd.GIDD")
        and _cfg_value(cfg, "model.name") == "tiny"
        and int(_cfg_value(cfg, "data.dataset_config.graph_N", -1)) == int(graph_n)
    )


def _has_checkpoint(run_dir: Path) -> bool:
    try:
        _resolve_ckpt_path(run_dir, CKPT_TYPE)
    except FileNotFoundError:
        return False
    return True


def discover_run_dirs(graph_ns=GRAPH_NS) -> dict[int, Path]:
    discovered = {}
    run_dirs = sorted({path.parents[1] for path in (ROOT / "outputs" / "brevo").glob("*/*/.hydra/config.yaml")})
    for graph_n in graph_ns:
        candidates = [run_dir for run_dir in run_dirs if _is_tiny_gidd_brevo_run(run_dir, graph_n)]
        candidates.sort(
            key=lambda run_dir: (
                _has_checkpoint(run_dir),
                (run_dir / "final_metrics.json").exists(),
                str(run_dir),
            ),
            reverse=True,
        )
        if candidates:
            discovered[graph_n] = candidates[0]
    return discovered


def load_model(run_dir: Path, ckpt_type: Literal["best", "last"] = "best"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_ckpt_path(run_dir, ckpt_type)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["hyper_parameters"]["config"]
    tokenizer = get_tokenizer(cfg)
    algo_cls = hydra.utils.get_class(cfg.algo._target_)
    model = algo_cls.load_from_checkpoint(
        ckpt_path,
        config=cfg,
        tokenizer=tokenizer,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, ckpt_path, cfg


def _brevo_dataset_config(cfg) -> dict:
    raw_cfg = _cfg_value(cfg, "data.dataset_config", {})
    container = OmegaConf.to_container(raw_cfg, resolve=True)
    return {key: value for key, value in dict(container).items() if value not in (None, "")}


def load_validation_prefixes(cfg, num_samples: int):
    records = list(
        islice(
            iter_brevo_split_records(
                dataset_config=_brevo_dataset_config(cfg),
                split="validation",
            ),
            num_samples,
        )
    )
    if not records:
        raise RuntimeError("Validation split produced no BREVO records.")
    return [record["prefixes"] for record in records], [record["completions"] for record in records]


def _tokenize_text_ids(tokenizer, values: list[str]) -> list[list[int]]:
    try:
        return tokenizer(
            values, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
    except (NotImplementedError, TypeError, AttributeError):
        tokenized = []
        for value in values:
            stripped = value.strip()
            tokenized.append([] if not stripped else [int(tok) for tok in stripped.split()])
        return tokenized


def build_eval_tensors(
    tokenizer,
    prefixes: list[str],
    completions: list[str],
    seq_len: int,
):
    prefix_tokens = _tokenize_text_ids(tokenizer, prefixes)
    completion_tokens = _tokenize_text_ids(tokenizer, completions)

    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(prefixes), seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prefixes), seq_len), dtype=torch.bool)
    accuracy_mask = torch.zeros((len(prefixes), seq_len), dtype=torch.bool)
    prefix_ids = []
    completion_ids = []
    for row_idx, (p_ids, c_ids) in enumerate(
        zip(prefix_tokens, completion_tokens, strict=True)
    ):
        p_ids = list(p_ids)
        c_ids = list(c_ids)

        # Match loaders.py for BREVO pretokenized examples: prefix + exact
        # completion span, then PAD with attention/accuracy masks set to 0.
        if len(p_ids) >= seq_len and c_ids:
            p_ids = p_ids[: max(seq_len - 1, 0)]
        else:
            p_ids = p_ids[:seq_len]
        c_ids = c_ids[: max(seq_len - len(p_ids), 0)]

        seq_ids = p_ids + c_ids
        prefix_ids.append(p_ids)
        completion_ids.append(c_ids)
        if seq_ids:
            input_ids[row_idx, : len(seq_ids)] = torch.as_tensor(
                seq_ids, dtype=torch.long
            )
            attention_mask[row_idx, : len(seq_ids)] = True
        if c_ids:
            start = len(p_ids)
            accuracy_mask[row_idx, start : start + len(c_ids)] = True
    fixed_mask = ~accuracy_mask
    return prefix_ids, completion_ids, input_ids, fixed_mask, attention_mask, accuracy_mask


def _ids_line(tokens: torch.Tensor, width: int) -> str:
    return " ".join(f"{int(tok):>{width}d}" for tok in tokens.tolist())


def _decode(tokenizer, tokens: torch.Tensor) -> str:
    return tokenizer.decode(tokens.detach().cpu().tolist(), skip_special_tokens=False)


def _answer_from_mask(tokens: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    return tokens.detach().cpu()[completion_mask.detach().cpu().to(dtype=torch.bool)]


def _line_from_ids(tokens: torch.Tensor) -> str:
    return " ".join(str(int(tok)) for tok in tokens.tolist())


def _brevo_topo_parser(cfg):
    multi = bool(_cfg_value(cfg, "data.dataset_config.multi_token", False))
    return TopoSortDepthStats.parse_tokens_multi if multi else TopoSortDepthStats.parse_tokens


def write_trace(
    path: Path,
    *,
    graph_n: int,
    run_dir: Path,
    ckpt_path: Path,
    sampler_name: str,
    trajectory: torch.Tensor,
    prefixes: list[str],
    completions: list[str],
    accuracy_mask: torch.Tensor,
    tokenizer,
    cfg,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    step_ids = list(range(0, trajectory.shape[0], max(1, int(STEP_EVERY))))
    if step_ids[-1] != trajectory.shape[0] - 1:
        step_ids.append(trajectory.shape[0] - 1)
    accuracy_mask = accuracy_mask.detach().cpu().to(dtype=torch.bool)
    final_tokens = trajectory[-1].detach().cpu()
    parser = _brevo_topo_parser(cfg)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"graph_N: {graph_n}\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"checkpoint: {ckpt_path}\n")
        f.write(f"sampler: {sampler_name}\n")
        f.write(f"num_steps: {trajectory.shape[0] - 1}\n")
        f.write(f"step_every: {STEP_EVERY}\n\n")

        for sample_idx in range(trajectory.shape[1]):
            sample_steps = [
                trajectory[step_idx, sample_idx].detach().cpu()
                for step_idx in step_ids
            ]
            id_width = max(
                1,
                max(
                    len(str(abs(int(tok))))
                    for tokens in sample_steps
                    for tok in tokens.tolist()
                ),
            )
            if any(int(tok) < 0 for tokens in sample_steps for tok in tokens.tolist()):
                id_width += 1

            f.write(f"sample: {sample_idx}\n")
            answer = _answer_from_mask(final_tokens[sample_idx], accuracy_mask[sample_idx])
            topo_ok, _, _ = parser(final_tokens[sample_idx].tolist())
            f.write(f"model_answer: {_line_from_ids(answer)}\n")
            f.write(f"brevo_topo_ok: {bool(topo_ok)}\n")
            f.write(f"prefix: {prefixes[sample_idx]}\n")
            f.write(f"target: {completions[sample_idx]}\n")
            for step_idx, tokens in zip(step_ids, sample_steps, strict=True):
                f.write(f"{step_idx:04d} ids: {_ids_line(tokens, id_width)}\n")
                if PRINT_TEXT:
                    f.write(f"{step_idx:04d} text: {_decode(tokenizer, tokens)}\n")
            f.write("\n")


@torch.no_grad()
def sample_run(graph_n: int, run_dir: Path):
    model, tokenizer, ckpt_path, cfg = load_model(run_dir, CKPT_TYPE)
    prefixes, completions = load_validation_prefixes(cfg, NUM_SAMPLES)
    _, _, input_tokens, fixed_mask, attention_mask, accuracy_mask = build_eval_tensors(
        tokenizer,
        prefixes,
        completions,
        model.num_tokens,
    )
    fixed_tokens = input_tokens
    fixed_tokens = fixed_tokens.to(model.device)
    fixed_mask = fixed_mask.to(model.device)
    attention_mask = attention_mask.to(model.device)

    num_steps = NUM_STEPS if NUM_STEPS is not None else int(cfg.sampling.steps)
    eps = float(EPS if EPS is not None else cfg.algo.t_eps)
    inject_bos = bool(getattr(cfg.sampling, "inject_bos", True))

    written = []
    for sampler_name, sampler_cls in SAMPLERS:
        sampler = sampler_cls(cfg)
        trajectory = sampler.generate(
            model=model,
            num_samples=len(prefixes),
            num_steps=num_steps,
            eps=eps,
            inject_bos=inject_bos,
            record_steps=True,
            fixed_tokens=fixed_tokens,
            fixed_mask=fixed_mask,
            attention_mask=attention_mask,
        )
        trace_path = OUT_DIR / f"graph_N_{graph_n}" / f"{sampler_name}.txt"
        write_trace(
            trace_path,
            graph_n=graph_n,
            run_dir=run_dir,
            ckpt_path=ckpt_path,
            sampler_name=sampler_name,
            trajectory=trajectory,
            prefixes=prefixes,
            completions=completions,
            accuracy_mask=accuracy_mask,
            tokenizer=tokenizer,
            cfg=cfg,
        )
        written.append(trace_path)
    return written


def run_all():
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    run_dirs = {**discover_run_dirs(), **RUN_DIRS}

    all_written = []
    for graph_n in GRAPH_NS:
        run_dir = run_dirs.get(graph_n)
        if run_dir is None:
            print(f"graph_N={graph_n}: no tiny GIDD run found.")
            continue
        try:
            print(f"graph_N={graph_n}: sampling {run_dir}")
            all_written.extend(sample_run(graph_n, run_dir))
        except FileNotFoundError as exc:
            if not SKIP_MISSING_CHECKPOINTS:
                raise
            print(f"graph_N={graph_n}: skipped: {exc}")

    print("\nTrace files:")
    for path in all_written:
        print(path)
    return all_written


if __name__ == "__main__":
    run_all()
