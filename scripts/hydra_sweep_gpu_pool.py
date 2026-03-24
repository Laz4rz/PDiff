#!/usr/bin/env python3
"""Run Hydra sweep points inside one node allocation with a GPU worker pool.

This script keeps a single source of truth in the Hydra config by reading
`hydra.sweeper.params` from the selected config and expanding it into runs.

Each run is executed as:
  python -m <module> --config-name <config> hydra.mode=RUN ...

with a unique `hydra.run.dir` and a per-process `CUDA_VISIBLE_DEVICES`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


DISPATCH_RESERVED_OVERRIDE_KEYS = {
    "hydra.mode",
    "hydra.run.dir",
    "data.cache_dir",
}

DISPATCH_REPRO_ENV_KEYS = [
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "DISCRETE_DIFFUSION_SCRATCH_DIR",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    "SLURM_TMPDIR",
    "WANDB_MODE",
    "WANDB_TOKEN_FILE",
    "WANDB_API_KEY",
]


def _split_csv(expr: str) -> list[str]:
    return [part.strip() for part in expr.split(",") if part.strip()]


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return f"{value.normalize():f}".rstrip("0").rstrip(".")


def _parse_range_expr(expr: str) -> list[str]:
    # Hydra-like range(start, stop[, step]) with stop exclusive.
    inside = expr[len("range(") : -1].strip()
    parts = _split_csv(inside)
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid range expression: {expr}")

    start = Decimal(parts[0])
    stop = Decimal(parts[1])
    step = Decimal(parts[2]) if len(parts) == 3 else Decimal(1)
    if step == 0:
        raise ValueError(f"Range step must be non-zero: {expr}")

    out: list[str] = []
    current = start
    if step > 0:
        while current < stop:
            out.append(_format_decimal(current))
            current += step
    else:
        while current > stop:
            out.append(_format_decimal(current))
            current += step
    return out


def _parse_axis_values(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("range(") and stripped.endswith(")"):
            return _parse_range_expr(stripped)
        return _split_csv(stripped)
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _parse_gpus(raw_gpus: str | None) -> list[str]:
    if raw_gpus is None:
        raw_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpus = [gpu.strip() for gpu in raw_gpus.replace(" ", ",").split(",") if gpu.strip()]
    if not gpus:
        return ["0"]
    return gpus


def _build_gpu_slots(gpus: list[str], max_parallel: int) -> list[str]:
    if max_parallel <= 0:
        raise ValueError("max_parallel must be >= 1")
    # Round-robin GPU assignment to allow oversubscription when
    # max_parallel > len(gpus), e.g. 8 workers on 4 GPUs.
    return [gpus[idx % len(gpus)] for idx in range(max_parallel)]


def _normalized_override_key(override: str) -> str:
    key = override.split("=", 1)[0]
    return key.lstrip("+~")


def _stable_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dispatch_repro_env(env: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key in DISPATCH_REPRO_ENV_KEYS:
        value = env.get(key)
        if value in (None, ""):
            continue
        if key == "WANDB_API_KEY":
            out[key] = "<redacted>"
        else:
            out[key] = value
    return out


def _inject_wandb_api_key(
    env: dict[str, str], configured_token_file: str | None = None
) -> None:
    if env.get("WANDB_API_KEY", "").strip():
        return

    token_file = env.get("WANDB_TOKEN_FILE") or configured_token_file
    if not token_file:
        print(
            "Warning: WANDB_API_KEY is unset and no token file is configured "
            "(set WANDB_TOKEN_FILE or pooled_slurm.wandb_token_file)."
        )
        return
    token_path = Path(token_file).expanduser()

    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(
            "Warning: WANDB_API_KEY is unset and token file could not be read: "
            f"{token_path} ({exc})"
        )
        return

    if not token:
        print(
            "Warning: WANDB_API_KEY is unset and token file is empty: "
            f"{token_path}"
        )
        return

    env["WANDB_API_KEY"] = token
    env["WANDB_TOKEN_FILE"] = str(token_path)
    print(f"Loaded WANDB_API_KEY from {token_path}.")


@dataclass
class ActiveRun:
    index: int
    gpu: str
    cmd: list[str]
    proc: subprocess.Popen[bytes]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand hydra.sweeper.params and run sweep points on a GPU pool."
    )
    parser.add_argument(
        "--config-name",
        default="gidd_star_graph_d3p3n50_constant_lr_sweep",
        help="Hydra config name under configs/ (default: gidd_star_graph_d3p3n50_constant_lr_sweep).",
    )
    parser.add_argument(
        "--module",
        default="discrete_diffusion",
        help="Python module entrypoint to run (default: discrete_diffusion).",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable for launched runs (default: current interpreter).",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids (default: CUDA_VISIBLE_DEVICES, else 0).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Maximum concurrent runs (default: number of GPUs). "
            "Can be larger than GPU count to oversubscribe in round-robin."
        ),
    )
    parser.add_argument(
        "--sweep-dir",
        default=None,
        help="Override run root directory. Defaults to hydra.sweep.dir + '/pooled-<timestamp>'.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override (repeatable), applied to compose and each run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of expanded runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing.",
    )
    args = parser.parse_args()
    user_override_keys = {_normalized_override_key(item) for item in args.override}
    forbidden_keys = sorted(user_override_keys & DISPATCH_RESERVED_OVERRIDE_KEYS)
    if forbidden_keys:
        raise SystemExit(
            "The pooled runner manages these overrides internally. "
            f"Remove from --override: {', '.join(forbidden_keys)}"
        )

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name=args.config_name,
            overrides=args.override,
            return_hydra_config=True,
        )
        frozen_task_cfg = compose(
            config_name=args.config_name,
            overrides=args.override,
            return_hydra_config=False,
        )

    sweeper_params = OmegaConf.to_container(
        cfg.hydra.sweeper.params, resolve=True
    ) or {}
    if not isinstance(sweeper_params, dict) or not sweeper_params:
        raise SystemExit("No hydra.sweeper.params found in config.")

    pooled_cfg = OmegaConf.to_container(cfg.get("pooled_slurm"), resolve=True) or {}
    configured_wandb_token_file: str | None = None
    configured_cache_root: str | None = None
    if isinstance(pooled_cfg, dict):
        token_value = pooled_cfg.get("wandb_token_file")
        if token_value not in (None, ""):
            configured_wandb_token_file = str(token_value)
        cache_root_value = pooled_cfg.get("cache_root")
        if cache_root_value not in (None, ""):
            configured_cache_root = str(cache_root_value)

    axes: list[tuple[str, list[str]]] = []
    for key, raw_value in sweeper_params.items():
        values = _parse_axis_values(raw_value)
        if not values:
            raise SystemExit(f"Sweep axis '{key}' has no values.")
        axes.append((str(key), values))

    keys = [k for k, _ in axes]
    value_lists = [v for _, v in axes]
    combinations = list(itertools.product(*value_lists))
    if args.limit is not None:
        combinations = combinations[: max(args.limit, 0)]

    if args.sweep_dir:
        sweep_root = Path(args.sweep_dir).expanduser()
    else:
        base_sweep_dir = Path(str(cfg.hydra.sweep.dir))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_root = base_sweep_dir / f"pooled-{timestamp}"
    if not sweep_root.is_absolute():
        sweep_root = (repo_root / sweep_root).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    base_cache_dir = Path(configured_cache_root or str(cfg.data.cache_dir)).expanduser()
    if not base_cache_dir.is_absolute():
        base_cache_dir = (repo_root / base_cache_dir).resolve()
    base_cache_dir = base_cache_dir / sweep_root.name
    tmp_base = Path(
        os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    ).expanduser()
    pool_tmp_root = tmp_base / f"pdiff-pool-{os.getpid()}"
    pool_tmp_root.mkdir(parents=True, exist_ok=True)

    frozen_config_name = f"_pooled_frozen_{args.config_name}"
    frozen_config_dir = pool_tmp_root / "config_snapshot"
    frozen_config_dir.mkdir(parents=True, exist_ok=True)
    frozen_config_path = frozen_config_dir / f"{frozen_config_name}.yaml"
    # Freeze task config + user overrides without forcing resolution of Hydra
    # runtime fields (e.g. hydra.job.num), which only exist during execution.
    OmegaConf.save(config=frozen_task_cfg, f=str(frozen_config_path), resolve=False)
    frozen_config_audit_path = sweep_root / "pool_config_snapshot.yaml"
    shutil.copy2(frozen_config_path, frozen_config_audit_path)

    gpus = _parse_gpus(args.gpus)
    max_parallel = args.max_parallel if args.max_parallel is not None else len(gpus)
    max_parallel = max(1, max_parallel)
    gpu_slots = _build_gpu_slots(gpus, max_parallel)

    runs: list[dict[str, Any]] = []
    for idx, combo in enumerate(combinations):
        run_cache_dir = base_cache_dir / f"pool-{idx:04d}"
        run_tmp_dir = pool_tmp_root / f"run-{idx:04d}"
        per_run_overrides = [
            f"data.cache_dir={run_cache_dir.as_posix()}",
            *[f"{k}={v}" for k, v in zip(keys, combo)],
        ]
        run_dir = sweep_root / f"{idx:04d}"
        cmd = [
            args.python_bin,
            "-m",
            args.module,
            "--config-path",
            frozen_config_dir.as_posix(),
            "--config-name",
            frozen_config_name,
            *per_run_overrides,
            "hydra.mode=RUN",
            f"hydra.run.dir={run_dir.as_posix()}",
        ]
        runs.append(
            {
                "index": idx,
                "axis_values": {k: v for k, v in zip(keys, combo)},
                "overrides": per_run_overrides,
                "run_dir": run_dir.as_posix(),
                "cache_dir": run_cache_dir.as_posix(),
                "tmp_dir": run_tmp_dir.as_posix(),
                "cmd": cmd,
            }
        )

    manifest = {
        "config_name": args.config_name,
        "module": args.module,
        "python_bin": args.python_bin,
        "base_overrides": args.override,
        "axes": {k: v for k, v in axes},
        "num_runs": len(runs),
        "gpus": gpus,
        "gpu_slots": gpu_slots,
        "max_parallel": max_parallel,
        "frozen_config_name": frozen_config_name,
        "frozen_config_path": frozen_config_path.as_posix(),
        "frozen_config_audit_path": frozen_config_audit_path.as_posix(),
        "runs": runs,
    }

    print(
        "Planned "
        f"{len(runs)} runs from axes {keys} | gpus={gpus} | "
        f"gpu_slots={gpu_slots} | max_parallel={max_parallel}"
    )
    print(f"Run root: {sweep_root}")
    print(f"Using frozen config snapshot: {frozen_config_path}")

    if args.dry_run:
        for run in runs:
            print(f"[dry-run #{run['index']}] {' '.join(run['cmd'])}")
        return 0

    repo_src = (repo_root / "src").resolve()
    base_env = os.environ.copy()
    _inject_wandb_api_key(base_env, configured_wandb_token_file)
    dispatch_env = _dispatch_repro_env(base_env)
    dispatch_signature = _stable_sha256(
        {
            "config_name": args.config_name,
            "module": args.module,
            "python_bin": args.python_bin,
            "base_overrides": args.override,
            "axes": keys,
            "dispatch_env": dispatch_env,
        }
    )
    base_env["PDIFF_DISPATCH_SIGNATURE"] = dispatch_signature
    manifest["dispatch_signature"] = dispatch_signature
    manifest["dispatch_env"] = dispatch_env
    with (sweep_root / "pool_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Dispatch signature: {dispatch_signature[:12]}")
    pending = runs.copy()
    available_gpus = gpu_slots.copy()
    active: list[ActiveRun] = []
    failed = False

    def _terminate_active() -> None:
        for ar in active:
            if ar.proc.poll() is None:
                ar.proc.terminate()
        for ar in active:
            if ar.proc.poll() is None:
                ar.proc.wait(timeout=30)

    def _sig_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        print(f"Received signal {signum}; terminating active runs...")
        _terminate_active()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    while pending or active:
        while pending and available_gpus and len(active) < max_parallel:
            run = pending.pop(0)
            gpu = available_gpus.pop(0)
            env = base_env.copy()
            # Run from source checkout without requiring `pip install -e .`.
            if repo_src.is_dir():
                existing_pythonpath = env.get("PYTHONPATH")
                if existing_pythonpath:
                    env["PYTHONPATH"] = f"{repo_src}{os.pathsep}{existing_pythonpath}"
                else:
                    env["PYTHONPATH"] = str(repo_src)
            run_tmp_dir = Path(run["tmp_dir"])
            run_tmp_dir.mkdir(parents=True, exist_ok=True)
            # Keep temporary multiprocess artifacts off NFS mounts.
            env["TMPDIR"] = run_tmp_dir.as_posix()
            env["TMP"] = run_tmp_dir.as_posix()
            env["TEMP"] = run_tmp_dir.as_posix()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PDIFF_RUN_INDEX"] = str(run["index"])
            proc = subprocess.Popen(run["cmd"], env=env, cwd=str(repo_root))
            active.append(
                ActiveRun(
                    index=run["index"],
                    gpu=gpu,
                    cmd=run["cmd"],
                    proc=proc,
                )
            )
            print(
                f"[start #{run['index']}] gpu={gpu} pid={proc.pid} "
                f"dispatch={dispatch_signature[:12]}"
            )

        still_active: list[ActiveRun] = []
        for ar in active:
            code = ar.proc.poll()
            if code is None:
                still_active.append(ar)
                continue
            available_gpus.append(ar.gpu)
            if code == 0:
                print(f"[done  #{ar.index}] gpu={ar.gpu} rc=0")
            else:
                print(f"[fail  #{ar.index}] gpu={ar.gpu} rc={code}")
                failed = True
        active = still_active

        if failed:
            print("Failure detected; terminating remaining active runs.")
            _terminate_active()
            return 1

        if pending or active:
            # Lightweight polling loop; no busy-spin.
            import time

            time.sleep(0.5)

    print("All pooled runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
