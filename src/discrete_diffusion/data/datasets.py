"""Specialized dataset builders re-used in the discrete diffusion loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import TypedDict
import urllib
import zipfile

import datasets
import fsspec
import numpy as np
import requests
from tqdm import tqdm

from .. import utils
from .generators.brevo import topsort_data

LOGGER = utils.get_logger(__name__)

__all__ = [
    "generate_synthetic_dataset",
    "get_star_graph_dataset",
    "get_brevo_dataset",
    "generate_prefix_dataset",
    "get_lambada_test_dataset",
    "get_text8_dataset",
]


class StarGraphDatasetConfig(TypedDict, total=False):
    data_dir: str | Path
    train_file: str | Path
    validation_file: str | Path


def get_star_graph_dataset(dataset_config: StarGraphDatasetConfig | None = None):
    repo_root = Path(__file__).resolve().parents[3]
    default_data_dir = repo_root / "data" / "star"
    config = dict(dataset_config or {})
    data_dir = Path(str(config.get("data_dir") or default_data_dir))
    if not data_dir.is_absolute():
        data_dir = repo_root / data_dir

    train_path = Path(
        str(config.get("train_file") or "deg_2_path_2_nodes_50_train_200000.txt")
    )
    if not train_path.is_absolute():
        train_path = data_dir / train_path

    validation_path = Path(
        str(config.get("validation_file") or "deg_2_path_2_nodes_50_test_20000.txt")
    )
    if not validation_path.is_absolute():
        validation_path = data_dir / validation_path

    def _read_star_graphs(path: Path) -> datasets.Dataset:
        if not path.exists():
            raise FileNotFoundError(f"Star graph dataset file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        prefixes = []
        completions = []
        for line in tqdm(lines, desc=f"Reading star graph dataset from {path}"):
            prefix, target = line.strip().split("=", maxsplit=1)
            prefixes.append(prefix + "=")
            completions.append(target)

        return datasets.Dataset.from_dict(
            {
                "prefixes": prefixes,
                "completions": completions,
            }
        )

    train_ds = _read_star_graphs(train_path)
    validation_ds = _read_star_graphs(validation_path)

    return datasets.DatasetDict(
        {
            "train": train_ds,
            "validation": validation_ds
        }
    )

class BrevoDatasetConfig(TypedDict, total=False):
    training_samples: int
    evaluation_samples: int
    graph_N: int
    enforce_n_for_training: bool
    multi_token: bool

def get_brevo_dataset(
    dataset_config: BrevoDatasetConfig | None = None,
) -> datasets.DatasetDict:
    config = dict(dataset_config or {})
    training_samples = int(config.get("training_samples", 200_000))
    evaluation_samples = int(config.get("evaluation_samples", 20_000))
    graph_N = int(config.get("graph_N", 110))
    enforce_n_for_training = bool(config.get("enforce_n_for_training", False))
    multi_token = bool(config.get("multi_token", False))

    if graph_N < 3:
        raise ValueError(f"`graph_N` must be >= 3, got {graph_N}")
    if training_samples <= 0 or evaluation_samples <= 0:
        raise ValueError("`training_samples` and `evaluation_samples` must be > 0")

    def _tokens_to_text(tokens: list[int], add_trailing_space: bool = False) -> str:
        if not tokens:
            return ""
        text = " ".join(str(token) for token in tokens)
        if add_trailing_space:
            text += " "
        return text

    def _build_split(num_samples: int, enforce_n: bool, split: str) -> datasets.Dataset:
        prefixes: list[str] = []
        completions: list[str] = []

        for _ in tqdm(range(num_samples), desc=f"Generating BREVO {split}"):
            sample = topsort_data(N=graph_N, multi=multi_token, enforce_n=enforce_n)
            token_ids = [int(token) for token in sample[0]]
            labels = [int(label) for label in sample["label"]]

            if len(token_ids) != len(labels):
                raise ValueError(
                    "BREVO sample has mismatched token and label lengths: "
                    f"{len(token_ids)} != {len(labels)}"
                )

            prefix_tokens = [
                token for token, label in zip(token_ids, labels, strict=True) if label == 0
            ]
            completion_tokens = [
                token for token, label in zip(token_ids, labels, strict=True) if label == 1
            ]

            prefixes.append(
                _tokens_to_text(prefix_tokens, add_trailing_space=bool(completion_tokens))
            )
            completions.append(_tokens_to_text(completion_tokens))

        return datasets.Dataset.from_dict(
            {
                "prefixes": prefixes,
                "completions": completions,
            }
        )

    train_ds = _build_split(
        num_samples=training_samples,
        enforce_n=enforce_n_for_training,
        split="train",
    )
    valid_ds = _build_split(
        num_samples=evaluation_samples,
        enforce_n=True,  # hardest-case eval: n = N
        split="validation",
    )

    return datasets.DatasetDict({"train": train_ds, "validation": valid_ds})


def _generate_synthetic_data(dataset_size, seq_len, vocab_size):
    dataset = np.zeros((dataset_size, seq_len), dtype=int)
    dataset[:, 0] = vocab_size - 2  # bos
    dataset[:, -1] = vocab_size - 1  # eos
    for i in range(dataset_size):
        temp = np.random.randint(vocab_size - 2)
        for j in reversed(range(1, seq_len - 1)):
            dataset[i, j] = temp
            if temp != 0:
                temp = temp // 4
            else:
                temp = np.random.randint(vocab_size - 2)
    return dataset


def generate_synthetic_dataset(
    train_dataset_size, validation_dataset_size, seq_len, vocab_size
):
    import torch
    np.random.seed(42)
    train_data = torch.from_numpy(
        _generate_synthetic_data(train_dataset_size, seq_len, vocab_size)
    )
    train_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": train_data,
            "attention_mask": torch.ones_like(train_data),
        }
    )
    train_dataset.set_format(type="torch")

    np.random.seed(41)
    validation_data = torch.from_numpy(
        _generate_synthetic_data(validation_dataset_size, seq_len, vocab_size)
    )
    validation_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": validation_data,
            "attention_mask": torch.ones_like(validation_data),
        }
    )
    validation_dataset.set_format(type="torch")

    return {
        "train": train_dataset,
        "validation": validation_dataset,
    }


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)
        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True, crop_train=False):
    """Adapted from D3PM text datasets."""
    url = "http://mattmahoney.net/dc/text8.zip"
    cache_dir = (
        f"{cache_dir}/text8" if not crop_train else f"{cache_dir}/text8-crop-train"
    )
    split_names = ["train", "validation", "test"]
    if not all(
        [utils.fsspec_exists(os.path.join(cache_dir, split)) for split in split_names]
    ):
        raw_cache_dir = os.path.join(cache_dir, "raw_data")
        if not all(
            [
                utils.fsspec_exists(os.path.join(raw_cache_dir, f"text8.{split}.txt"))
                for split in split_names
            ]
        ):
            if not utils.fsspec_exists(os.path.join(raw_cache_dir, "text8.zip")):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info("Downloading text8 from URL %s.", url)
                with (
                    urllib.request.urlopen(url) as in_stream,
                    open(os.path.join(raw_cache_dir, "text8.zip"), "wb") as out_file,
                ):
                    shutil.copyfileobj(in_stream, out_file)
            with fsspec.open(os.path.join(raw_cache_dir, "text8.zip"), "rb") as f:
                rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")
            splits = {
                "train": rawdata[:90000000],
                "validation": rawdata[90000000:95000000],
                "test": rawdata[95000000:],
            }
            for split, data in splits.items():
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "w") as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
                with fsspec.open(_path, "r") as f:
                    splits[split] = f.read()

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        dataset_dict = {}
        for k, v in splits.items():
            chunk_size = (
                2 * max_seq_length if (k == "train" and crop_train) else max_seq_length
            )
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({"text": text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)
    return dataset


def generate_prefix_dataset(samples: int|None = 2048) -> datasets.DatasetDict:
    """
        if samples is None, generates a single repetition dataset
        otherwise repeats the prompts to reach the desired number of samples
    """
    prefixes = [
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
    completions = [
        "Paris",
        "Berlin",
        "Rome",
        "Madrid",
        "Tokyo",
        "Ottawa",
        "Canberra",
        "4",
        "30",
        "5",
        "cold",
        "down",
        "blue",
        "Monday",
        "Jupiter",
        "Celsius",
        "Au",
        "Portuguese",
        "Orwell",
        "9",
        "D",
        "went",
        "kitten",
        "triangle",
    ]
    prefixes = prefixes[:]
    completions = completions[:]
    if samples is None:
        repeat_factor = 1
    else:
        repeat_factor = samples // len(prefixes)
        if repeat_factor < 1:
            raise ValueError("Not enough samples to generate the desired number of repetitions")
    prefixes = prefixes * repeat_factor
    completions = completions * repeat_factor
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"prefixes": prefixes, "completions": completions}
            ),
            "validation": datasets.Dataset.from_dict(
                {"prefixes": prefixes, "completions": completions}
            )
        }
    )
