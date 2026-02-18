"""Specialized dataset builders re-used in the discrete diffusion loader."""

from __future__ import annotations

import json
import os
import random
import shutil
import string
import urllib
import zipfile

import datasets
import fsspec
import numpy as np
import requests
import torch

from .. import utils

LOGGER = utils.get_logger(__name__)

__all__ = [
    "generate_synthetic_dataset",
    "generate_star_graph_dataset",
    "get_lambada_test_dataset",
    "get_text8_dataset",
]


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


def _build_star_graph_prompt(
    *,
    rng: random.Random,
    context_window: int,
    nvertices: int,
    ndistractors: int,
    max_attention: int,
) -> tuple[str, str]:
    if context_window < 1:
        raise ValueError("star_graph context_window must be >= 1")
    if nvertices < 2:
        raise ValueError("star_graph nvertices must be >= 2")
    if nvertices > len(string.ascii_uppercase):
        raise ValueError(
            f"star_graph nvertices must be <= {len(string.ascii_uppercase)}"
        )
    if ndistractors < 0:
        raise ValueError("star_graph ndistractors must be >= 0")

    vertices = list(string.ascii_uppercase[:nvertices])
    graph = {}
    for vertex in vertices:
        targets = [candidate for candidate in vertices if candidate != vertex]
        graph[vertex] = rng.choice(targets)

    query_node = rng.choice(vertices)
    answer_node = graph[query_node]
    edges = [(query_node, answer_node)]
    for _ in range(context_window - 1):
        vertex = rng.choice(vertices)
        edges.append((vertex, graph[vertex]))

    if max_attention <= 0:
        selected_edges = edges[:1]
    elif max_attention >= len(edges):
        selected_edges = list(edges)
    else:
        selected_edges = rng.sample(edges, max_attention)
    rng.shuffle(selected_edges)

    max_num_distractors = min(ndistractors, len(vertices) - 1)
    distractor_pool = [vertex for vertex in vertices if vertex != answer_node]
    distractors = rng.sample(distractor_pool, k=max_num_distractors)
    options = [answer_node] + distractors
    rng.shuffle(options)

    selected_edges_str = ", ".join(
        [f"{{{source}, {target}}}" for source, target in selected_edges]
    )
    options_str = ", ".join(options)
    prompt = (
        "Given the following list of directed edges in a graph:\n"
        f"[{selected_edges_str}]\n"
        "And the query node:\n"
        f"{query_node}\n"
        "What is the correct next node in the sequence among the "
        f"following options: [{options_str}]?"
    )
    return prompt, answer_node


def _generate_star_graph_texts(
    dataset_size: int,
    *,
    seed: int,
    context_window: int,
    nvertices: int,
    ndistractors: int,
    max_attention: int,
    instruction: str,
) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(dataset_size):
        prompt, answer = _build_star_graph_prompt(
            rng=rng,
            context_window=context_window,
            nvertices=nvertices,
            ndistractors=ndistractors,
            max_attention=max_attention,
        )
        text = f"Instruction: {instruction}\nInput: {prompt}\nOutput: {answer}"
        texts.append(text)
    return texts


def generate_star_graph_dataset(
    train_dataset_size: int,
    validation_dataset_size: int,
    *,
    context_window: int,
    nvertices: int,
    ndistractors: int,
    max_attention: int,
    instruction: str,
    train_seed: int,
    validation_seed: int,
):
    """Generate a Star-Graph dataset (ported from Next-Token-Failures, MIT)."""
    train_texts = _generate_star_graph_texts(
        train_dataset_size,
        seed=train_seed,
        context_window=context_window,
        nvertices=nvertices,
        ndistractors=ndistractors,
        max_attention=max_attention,
        instruction=instruction,
    )
    validation_texts = _generate_star_graph_texts(
        validation_dataset_size,
        seed=validation_seed,
        context_window=context_window,
        nvertices=nvertices,
        ndistractors=ndistractors,
        max_attention=max_attention,
        instruction=instruction,
    )
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict({"text": train_texts}),
            "validation": datasets.Dataset.from_dict({"text": validation_texts}),
        }
    )


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


def generate_prefix_dataset():
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
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {"prefix": prefixes, "completions": completions}
            ),
            "validation": datasets.Dataset.from_dict({"text": []}),
        }
    )
