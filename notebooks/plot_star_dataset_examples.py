#!/usr/bin/env python3
"""Plot one star-graph example from each dataset variant."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install it first, e.g. `pip install matplotlib`."
    ) from exc

try:
    import networkx as nx
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: networkx. Install it first, e.g. `pip install networkx`."
    ) from exc


SPEC_PATTERN = re.compile(
    r"^(deg_[^_]+_path_[^_]+_nodes_[^_]+(?:_reverse_[^_]+)?)_"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot one graph sample from each star dataset variant."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "star",
        help="Directory containing star dataset txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "star_dataset_examples.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Line index to plot from each dataset file (wraps via modulo).",
    )
    parser.add_argument(
        "--prefer-split",
        choices=("train", "test"),
        default="train",
        help="Prefer using train or test file for each dataset variant.",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=42,
        help="Random seed for spring-layout node placement.",
    )
    return parser.parse_args()


def spec_from_filename(filename: str) -> str | None:
    match = SPEC_PATTERN.match(filename)
    if not match:
        return None
    return match.group(1)


def collect_dataset_files(data_dir: Path) -> Dict[str, Dict[str, Path]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    datasets_by_spec: Dict[str, Dict[str, Path]] = {}
    for path in sorted(data_dir.glob("*.txt")):
        spec = spec_from_filename(path.name)
        if spec is None:
            continue
        bucket = datasets_by_spec.setdefault(spec, {})
        if "_train" in path.name and "train" not in bucket:
            bucket["train"] = path
        if "_test" in path.name and "test" not in bucket:
            bucket["test"] = path
    return datasets_by_spec


def parse_star_line(line: str) -> Tuple[List[Tuple[int, int]], Tuple[int, int], List[int]]:
    text = line.strip()
    if "=" not in text:
        raise ValueError(f"Line does not contain '=' separator: {text!r}")
    prefix, completion = text.split("=", maxsplit=1)

    if "/" not in prefix:
        raise ValueError(f"Prefix does not contain '/' separator: {prefix!r}")
    edge_blob, query_blob = prefix.split("/", maxsplit=1)

    edges: List[Tuple[int, int]] = []
    if edge_blob:
        for token in edge_blob.split("|"):
            if not token:
                continue
            u, v = token.split(",", maxsplit=1)
            edges.append((int(u), int(v)))

    query_src, query_dst = [int(x) for x in query_blob.split(",", maxsplit=1)]
    path_nodes = [int(x) for x in completion.split(",") if x]
    return edges, (query_src, query_dst), path_nodes


def choose_sample_line(dataset_file: Path, sample_index: int) -> Tuple[str, int, int]:
    with dataset_file.open("r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]

    if not lines:
        raise ValueError(f"No non-empty lines found in {dataset_file}")

    idx = sample_index % len(lines)
    return lines[idx], idx, len(lines)


def choose_file(split_files: Dict[str, Path], prefer_split: str) -> Path:
    preferred = split_files.get(prefer_split)
    if preferred is not None:
        return preferred
    fallback = "test" if prefer_split == "train" else "train"
    alt = split_files.get(fallback)
    if alt is not None:
        return alt
    raise ValueError(f"Dataset variant missing train/test file: {split_files}")


def plot_variant(
    ax,
    spec: str,
    dataset_file: Path,
    sample_index: int,
    layout_seed: int,
) -> None:
    line, resolved_index, total_rows = choose_sample_line(dataset_file, sample_index)
    edges, (src, dst), path_nodes = parse_star_line(line)

    graph = nx.Graph()
    graph.add_edges_from(edges)
    if not graph.nodes:
        graph.add_node(src)
        graph.add_node(dst)

    path_edges = list(zip(path_nodes[:-1], path_nodes[1:])) if len(path_nodes) > 1 else []

    pos = nx.spring_layout(graph, seed=layout_seed)

    all_nodes = list(graph.nodes())
    path_node_set = set(path_nodes)
    node_colors = []
    for node in all_nodes:
        if node == src:
            node_colors.append("#2ca02c")
        elif node == dst:
            node_colors.append("#d62728")
        elif node in path_node_set:
            node_colors.append("#ff7f0e")
        else:
            node_colors.append("#8c8c8c")

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#b5b5b5", width=1.4)
    if path_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=path_edges,
            edge_color="#ff7f0e",
            width=3.2,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=220,
        node_color=node_colors,
        linewidths=0.4,
        edgecolors="#222222",
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)

    ax.set_title(
        f"{spec}\n{dataset_file.name} | row {resolved_index}/{total_rows - 1}",
        fontsize=9,
    )
    ax.text(
        0.01,
        0.01,
        f"query: {src}->{dst}\npath: {','.join(map(str, path_nodes))}",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "pad": 2, "edgecolor": "none"},
    )
    ax.set_axis_off()


def main() -> None:
    args = parse_args()

    datasets_by_spec = collect_dataset_files(args.data_dir)
    specs = sorted(datasets_by_spec)
    if not specs:
        raise SystemExit(f"No star dataset files found in {args.data_dir}")

    cols = min(3, len(specs))
    rows = math.ceil(len(specs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.3 * cols, 4.9 * rows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, spec in enumerate(specs):
        split_files = datasets_by_spec[spec]
        dataset_file = choose_file(split_files, args.prefer_split)
        plot_variant(
            axes_list[i],
            spec=spec,
            dataset_file=dataset_file,
            sample_index=args.sample_index,
            layout_seed=args.layout_seed,
        )

    for j in range(len(specs), len(axes_list)):
        axes_list[j].set_axis_off()

    fig.suptitle(
        "Star Dataset Variants: One Example Graph Each",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
