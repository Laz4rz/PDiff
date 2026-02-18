"""Callback for exporting final training metrics to JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch


class FinalMetricsWriter(L.Callback):
    """Write final scalar metrics to a JSON file at the end of training."""

    def __init__(
        self, enabled: bool = True, save_path: str = "${cwd:}/final_metrics.json"
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.save_path = Path(save_path)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if not self.enabled or not trainer.is_global_zero:
            return

        merged_metrics = self._merge_metrics(
            trainer.callback_metrics, trainer.logged_metrics
        )

        payload: dict[str, Any] = {
            "global_step": int(trainer.global_step),
            "current_epoch": int(trainer.current_epoch),
            "final_metrics": self._group_metrics(merged_metrics),
            "checkpoints": self._checkpoint_summaries(trainer),
            # Raw dicts are kept for debugging; prefer "final_metrics" for reporting.
            "raw_callback_metrics": self._to_jsonable_dict(trainer.callback_metrics),
            "raw_logged_metrics": self._to_jsonable_dict(trainer.logged_metrics),
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)

    def _merge_metrics(
        self, callback_metrics: dict[str, Any], logged_metrics: dict[str, Any]
    ) -> dict[str, float]:
        merged = self._scalar_metrics(callback_metrics)
        for key, value in self._scalar_metrics(logged_metrics).items():
            merged.setdefault(key, value)
        return merged

    def _scalar_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        scalars: dict[str, float] = {}
        for key, value in metrics.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                scalars[str(key)] = scalar
        return scalars

    def _to_scalar(self, value: Any) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _group_metrics(self, metrics: dict[str, float]) -> dict[str, dict[str, float]]:
        grouped = {
            "train": {},
            "val": {},
            "trainer": {},
            "other": {},
        }
        for key, value in sorted(metrics.items()):
            if key.startswith("train/"):
                grouped["train"][key] = value
            elif key.startswith("val/"):
                grouped["val"][key] = value
            elif key.startswith("trainer/"):
                grouped["trainer"][key] = value
            else:
                grouped["other"][key] = value
        return grouped

    def _checkpoint_summaries(self, trainer: L.Trainer) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for callback in trainer.callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            summaries.append(
                {
                    "monitor": callback.monitor,
                    "mode": callback.mode,
                    "best_model_score": self._to_jsonable(callback.best_model_score),
                    "best_model_path": callback.best_model_path or None,
                    "last_model_path": callback.last_model_path or None,
                    "dirpath": str(callback.dirpath)
                    if callback.dirpath is not None
                    else None,
                }
            )
        return summaries

    def _to_jsonable_dict(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {str(key): self._to_jsonable(value) for key, value in metrics.items()}

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        return str(value)
