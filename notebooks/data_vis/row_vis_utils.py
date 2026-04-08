"""Shared row visualization helpers for dataset notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

__all__ = ["plot_token_masks_row"]


def _as_list_1d(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _display_label(token_id: int, tokenizer=None, label_mode: str = "ids") -> str:
    if label_mode == "ids" or tokenizer is None:
        return str(token_id)
    if label_mode != "chars":
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    try:
        label = tokenizer.convert_ids_to_tokens(int(token_id))
    except Exception:
        return str(token_id)

    if label == " ":
        return "<sp>"
    if label == "\n":
        return "<nl>"
    if label == "\t":
        return "<tab>"
    return str(label)


def plot_token_masks_row(
    sample,
    *,
    tokenizer=None,
    label_mode: str = "ids",
    max_tokens: int | None = 64,
    mask_order: Sequence[str] = (
        "attention_mask",
        "loss_mask",
        "accuracy_mask",
        "noise_mask",
    ),
    mask_colors: dict[str, str] | None = None,
    title: str | None = None,
):
    """Plot one token row with per-mask color bands inside each cell."""
    if mask_colors is None:
        mask_colors = {
            "attention_mask": "#4e79a7",
            "loss_mask": "#e15759",
            "accuracy_mask": "#59a14f",
            "noise_mask": "#f28e2b",
        }

    input_ids = _as_list_1d(sample["input_ids"])
    total_tokens = len(input_ids)
    shown_tokens = min(total_tokens, max_tokens) if max_tokens is not None else total_tokens
    input_ids = input_ids[:shown_tokens]

    labels = [
        _display_label(tok, tokenizer=tokenizer, label_mode=label_mode)
        for tok in input_ids
    ]

    active_masks = [name for name in mask_order if name in sample]
    mask_values = {name: _as_list_1d(sample[name])[:shown_tokens] for name in active_masks}

    fig_width = max(10, shown_tokens * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, 2.8))

    cell_h = 1.0
    band_h = 0.25
    num_masks = max(len(active_masks), 1)
    band_w = 1.0 / num_masks

    for col, label in enumerate(labels):
        ax.add_patch(
            Rectangle(
                (col, 0.0),
                1.0,
                cell_h,
                facecolor="white",
                edgecolor="#2f2f2f",
                linewidth=0.8,
            )
        )
        ax.text(col + 0.5, 0.62, label, ha="center", va="center", fontsize=8)

        for m_idx, mask_name in enumerate(active_masks):
            is_on = int(mask_values[mask_name][col]) == 1
            color = mask_colors.get(mask_name, "#999999") if is_on else "#f2f2f2"
            ax.add_patch(
                Rectangle(
                    (col + m_idx * band_w, 0.0),
                    band_w,
                    band_h,
                    facecolor=color,
                    edgecolor="#ffffff",
                    linewidth=0.5,
                )
            )

    ax.set_xlim(0, shown_tokens)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([i + 0.5 for i in range(shown_tokens)])
    ax.set_xticklabels([str(i) for i in range(shown_tokens)], fontsize=7, rotation=90)
    ax.set_yticks([])
    ax.set_xlabel("Token position")
    if title is None:
        label_desc = "chars" if label_mode == "chars" else "token ids"
        title = f"Token row with {label_desc} labels"
    ax.set_title(f"{title} (showing {shown_tokens}/{total_tokens} tokens)")

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [
        Patch(facecolor=mask_colors.get(name, "#999999"), edgecolor="none", label=name)
        for name in active_masks
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=min(len(handles), 4),
            frameon=False,
        )

    plt.tight_layout()
    return fig, ax

