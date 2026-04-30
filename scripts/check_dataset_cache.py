#!/usr/bin/env python3
"""Inspect cached tokenized datasets and dry-run expected cache paths.

Usage examples:
  python scripts/check_dataset_cache.py
  python scripts/check_dataset_cache.py --config-name gidd_brevo
  python scripts/check_dataset_cache.py --config-name gidd_brevo --expected-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SRC_DIR.as_posix())

from discrete_diffusion.data.cached_synthetic_builders import (
    get_cached_synthetic_builder,
)


def _default_cache_root() -> Path:
    if "DISCRETE_DIFFUSION_SCRATCH_DIR" in os.environ:
        return Path(os.environ["DISCRETE_DIFFUSION_SCRATCH_DIR"]).expanduser()
    return Path.home() / ".cache" / "discrete_diffusion"


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            total += (root_path / name).stat().st_size
    return total


def _list_cached_artifacts(cache_root: Path) -> list[tuple[str, Path, int, str]]:
    rows: list[tuple[str, Path, int, str]] = []
    if not cache_root.exists():
        return rows
    for dataset_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        for artifact in sorted(dataset_dir.glob("*.dat")):
            if not artifact.is_dir():
                continue
            size = _dir_size_bytes(artifact)
            mtime = dt.datetime.fromtimestamp(artifact.stat().st_mtime).isoformat(
                timespec="seconds"
            )
            rows.append((dataset_dir.name, artifact, size, mtime))
    return rows


def _cache_filename(
    dataset_name: str,
    mode: str,
    block_size: int,
    wrap: bool,
    insert_eos: bool,
    insert_special_tokens: bool,
    min_length: int,
    chunking: str,
    dataset_config: dict[str, Any] | None,
    tokenize: bool,
) -> str:
    chunking_mode = chunking.lower()
    if chunking_mode not in {"none", "double_newline"}:
        raise ValueError(f"Unsupported chunking mode: {chunking_mode}")
    if wrap and chunking_mode != "none":
        raise ValueError("Delimiter-based chunking only applies when wrap=False.")

    eos_tag = ""
    if not insert_eos:
        eos_tag += "_eosFalse"
    if not insert_special_tokens:
        eos_tag += "_specialFalse"
    min_len_tag = f"_min{min_length}" if (min_length and not wrap) else ""
    chunk_tag = "_flexchunk" if (not wrap and chunking_mode != "none") else ""

    if wrap:
        filename = f"{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat"
    else:
        filename = (
            f"{dataset_name}_{mode}_bs{block_size}_unwrapped"
            f"{chunk_tag}{eos_tag}{min_len_tag}.dat"
        )

    builder = get_cached_synthetic_builder(dataset_name, dataset_config, tokenize=tokenize)
    if builder is not None:
        filename = builder.cache_filename(filename)
    return filename


def _must_select(cfg, key: str):
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise KeyError(f"Missing required config key: {key}")
    return value


def _expected_paths_for_config(config_name: str) -> list[tuple[str, Path]]:
    config_dir = (Path(__file__).resolve().parents[1] / "configs").as_posix()
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name)

    data_train = str(_must_select(cfg, "data.train"))
    data_valid = str(_must_select(cfg, "data.valid"))
    tokenizer_name = str(_must_select(cfg, "data.tokenizer_name_or_path"))
    cache_dir = Path(str(_must_select(cfg, "data.cache_dir"))).expanduser()
    dataset_config = OmegaConf.select(cfg, "data.dataset_config")

    default_chunking = str(_must_select(cfg, "data.chunking"))
    train_chunking = (
        str(OmegaConf.select(cfg, "data.train_chunking"))
        if OmegaConf.select(cfg, "data.train_chunking") is not None
        else default_chunking
    )
    valid_chunking = (
        str(OmegaConf.select(cfg, "data.valid_chunking"))
        if OmegaConf.select(cfg, "data.valid_chunking") is not None
        else default_chunking
    )

    default_min = int(_must_select(cfg, "data.min_length"))
    train_min = (
        int(OmegaConf.select(cfg, "data.train_min_length"))
        if OmegaConf.select(cfg, "data.train_min_length") is not None
        else default_min
    )
    valid_min = (
        int(OmegaConf.select(cfg, "data.valid_min_length"))
        if OmegaConf.select(cfg, "data.valid_min_length") is not None
        else default_min
    )

    model_length = int(_must_select(cfg, "model.length"))
    wrap = bool(_must_select(cfg, "data.wrap"))
    insert_train_eos = bool(_must_select(cfg, "data.insert_train_eos"))
    insert_valid_eos = bool(_must_select(cfg, "data.insert_valid_eos"))
    insert_train_special = bool(_must_select(cfg, "data.insert_train_special"))
    insert_valid_special = bool(_must_select(cfg, "data.insert_valid_special"))

    train_tokenize = not (
        data_train == "brevo" and tokenizer_name == "brevo-dummy"
    )
    valid_tokenize = not (
        data_valid == "brevo" and tokenizer_name == "brevo-dummy"
    )

    valid_mode = "test" if data_valid in {"text8", "lm1b", "ag_news"} else "validation"

    train_filename = _cache_filename(
        dataset_name=data_train,
        mode="train",
        block_size=model_length,
        wrap=wrap,
        insert_eos=insert_train_eos,
        insert_special_tokens=insert_train_special,
        min_length=train_min,
        chunking=str(train_chunking),
        dataset_config=dataset_config,
        tokenize=train_tokenize,
    )
    valid_filename = _cache_filename(
        dataset_name=data_valid,
        mode=valid_mode,
        block_size=model_length,
        wrap=wrap,
        insert_eos=insert_valid_eos,
        insert_special_tokens=insert_valid_special,
        min_length=valid_min,
        chunking=str(valid_chunking),
        dataset_config=dataset_config,
        tokenize=valid_tokenize,
    )

    return [
        ("train", cache_dir / train_filename),
        ("valid", cache_dir / valid_filename),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List cached dataset artifacts and dry-run expected cache paths."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=_default_cache_root(),
        help="Root cache directory (default: DISCRETE_DIFFUSION_SCRATCH_DIR or ~/.cache/discrete_diffusion).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name to compute expected train/valid cache paths.",
    )
    parser.add_argument(
        "--expected-only",
        action="store_true",
        help="Only print expected cache paths for --config-name.",
    )
    args = parser.parse_args()

    cache_root = args.cache_root.expanduser()

    if not args.expected_only:
        rows = _list_cached_artifacts(cache_root)
        print(f"Cache root: {cache_root}")
        if not rows:
            print("No cached .dat dataset artifacts found.")
        else:
            print(f"Found {len(rows)} cached artifact(s):")
            for dataset_name, path, size, mtime in rows:
                print(
                    f"- {dataset_name:12s} {_human_bytes(size):>9s}  {mtime}  {path}"
                )

    if args.config_name is not None:
        print("")
        print(f"Expected cache paths for config: {args.config_name}")
        expected = _expected_paths_for_config(args.config_name)
        for split_name, path in expected:
            exists = path.exists()
            status = "HIT" if exists else "MISS"
            print(f"- {split_name:5s} [{status}] {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
