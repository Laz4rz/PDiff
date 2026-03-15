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
import itertools
import json
import os
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
        default="gidd_star_graph_constant_lr_sweep",
        help="Hydra config name under configs/ (default: gidd_star_graph_constant_lr_sweep).",
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
        help="Maximum concurrent runs (default: number of GPUs).",
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

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name=args.config_name,
            overrides=args.override,
            return_hydra_config=True,
        )

    sweeper_params = OmegaConf.to_container(
        cfg.hydra.sweeper.params, resolve=True
    ) or {}
    if not isinstance(sweeper_params, dict) or not sweeper_params:
        raise SystemExit("No hydra.sweeper.params found in config.")

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

    gpus = _parse_gpus(args.gpus)
    max_parallel = args.max_parallel if args.max_parallel is not None else len(gpus)
    max_parallel = max(1, min(max_parallel, len(gpus)))

    runs: list[dict[str, Any]] = []
    for idx, combo in enumerate(combinations):
        per_run_overrides = [f"{k}={v}" for k, v in zip(keys, combo)]
        run_dir = sweep_root / f"{idx:04d}"
        cmd = [
            args.python_bin,
            "-m",
            args.module,
            "--config-name",
            args.config_name,
            "hydra.mode=RUN",
            f"hydra.run.dir={run_dir.as_posix()}",
            *args.override,
            *per_run_overrides,
        ]
        runs.append(
            {
                "index": idx,
                "overrides": per_run_overrides,
                "run_dir": run_dir.as_posix(),
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
        "max_parallel": max_parallel,
        "runs": runs,
    }
    with (sweep_root / "pool_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(
        f"Planned {len(runs)} runs from axes {keys} | gpus={gpus} | max_parallel={max_parallel}"
    )
    print(f"Run root: {sweep_root}")

    if args.dry_run:
        for run in runs:
            print(f"[dry-run #{run['index']}] {' '.join(run['cmd'])}")
        return 0

    pending = runs.copy()
    available_gpus = gpus.copy()
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
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen(run["cmd"], env=env)
            active.append(
                ActiveRun(
                    index=run["index"],
                    gpu=gpu,
                    cmd=run["cmd"],
                    proc=proc,
                )
            )
            print(f"[start #{run['index']}] gpu={gpu} pid={proc.pid}")

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
