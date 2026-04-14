"""Shared cached-build abstractions for synthetic datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping

from .. import utils
from .datasets import iter_brevo_split_records

LOGGER = utils.get_logger(__name__)


class CachedSyntheticDatasetBuilder(ABC):
    """Base class for generated datasets cached as tokenized Arrow splits."""

    dataset_name: str

    def __init__(self, dataset_config: Mapping[str, object], *, tokenize: bool):
        self.dataset_config = dict(dataset_config)
        self.tokenize = bool(tokenize)
        self._generation_config = self._build_generation_config()
        self.cache_key = hashlib.sha1(
            json.dumps(self._generation_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]

    @abstractmethod
    def _build_generation_config(self) -> dict[str, object]:
        """Return the config payload used for cache-key hashing."""

    @abstractmethod
    def split_size(self, split: str) -> int:
        """Return number of records for the requested split."""

    @abstractmethod
    def iter_split_records(self, split: str) -> Iterator[dict[str, object]]:
        """Yield raw records for a split."""

    @property
    def force_regenerate(self) -> bool:
        return bool(self.dataset_config["force_regenerate"])

    @property
    def cached_streaming(self) -> bool:
        return bool(self.dataset_config["cached_streaming"])

    @property
    def cached_streaming_num_shards(self) -> int:
        return int(self.dataset_config["cached_streaming_num_shards"])

    def cache_filename(self, filename: str) -> str:
        return filename.replace(".dat", f"_cfg{self.cache_key}.dat")

    def maybe_remove_existing_cache(self, cache_path: str) -> None:
        if not self.force_regenerate:
            return
        if not os.path.exists(cache_path):
            return
        LOGGER.info(
            "%s force_regenerate=True; removing existing cache at: %s",
            self.dataset_name.upper(),
            cache_path,
        )
        if os.path.isdir(cache_path):
            shutil.rmtree(cache_path)
        else:
            os.remove(cache_path)

    def should_reuse_cache(self) -> bool:
        return not self.force_regenerate

    def should_use_cached_streaming(self, *, streaming: bool) -> bool:
        return streaming and self.cached_streaming and self.should_reuse_cache()


class BrevoCachedDatasetBuilder(CachedSyntheticDatasetBuilder):
    """Cached-build policy for BREVO synthetic dataset."""

    dataset_name = "brevo"

    def _build_generation_config(self) -> dict[str, object]:
        return {
            "training_samples": int(self.dataset_config["training_samples"]),
            "evaluation_samples": int(self.dataset_config["evaluation_samples"]),
            "graph_N": int(self.dataset_config["graph_N"]),
            "enforce_n_for_training": bool(self.dataset_config["enforce_n_for_training"]),
            "multi_token": bool(self.dataset_config["multi_token"]),
            "tokenize": self.tokenize,
        }

    def split_size(self, split: str) -> int:
        if split == "train":
            return int(self.dataset_config["training_samples"])
        if split == "validation":
            return int(self.dataset_config["evaluation_samples"])
        raise ValueError(
            f"Unsupported BREVO split {split!r}. Expected one of: train, validation."
        )

    def iter_split_records(self, split: str) -> Iterator[dict[str, object]]:
        return iter_brevo_split_records(dataset_config=self.dataset_config, split=split)


def get_cached_synthetic_builder(
    dataset_name: str, dataset_config: Mapping[str, object] | None, *, tokenize: bool
) -> CachedSyntheticDatasetBuilder | None:
    """Factory for synthetic cached-build builders."""

    if dataset_name == "brevo":
        if dataset_config is None:
            raise ValueError("BREVO dataset requires `data.dataset_config`.")
        return BrevoCachedDatasetBuilder(dataset_config, tokenize=tokenize)
    return None

