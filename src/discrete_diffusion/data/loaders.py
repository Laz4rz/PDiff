"""Top-level loader API for discrete diffusion training."""

from __future__ import annotations

import functools
import os
import shutil
from typing import Optional

import datasets
import numpy as np
import tokenizers
import torch
import transformers

from .. import utils
from .datasets import (
    get_brevo_dataset,
    generate_prefix_dataset,
    generate_synthetic_dataset,
    get_lambada_test_dataset,
    get_text8_dataset,
    get_star_graph_dataset,
)
from .cached_synthetic_builders import get_cached_synthetic_builder
from .processing import (
    _apply_detokenizer,
    _group_texts,
    lm1b_detokenizer,
    lambada_detokenizer,
    ptb_detokenizer,
    scientific_papers_detokenizer,
    wt_detokenizer,
)
from .tokenizers import (
    AsciiCharTokenizer,
    BrevoDummyTokenizer,
    SyntheticTokenizer,
    Text8Tokenizer,
)
from .flex_chunking import chunk_documents

LOGGER = utils.get_logger(__name__)
LOAD_FROM_CACHE = False

__all__ = [
    "get_tokenizer",
    "get_dataset",
    "get_dataloaders",
]


def _with_torch_collatable_format(dataset):
    # Avoid Hugging Face formatters importing optional torchvision components.
    return dataset


def _collate_examples(examples):
    batch = {}
    for key in examples[0]:
        values = [example[key] for example in examples]
        first = values[0]
        if torch.is_tensor(first):
            batch[key] = torch.stack(values)
        elif isinstance(first, np.ndarray):
            batch[key] = torch.from_numpy(np.stack(values))
        else:
            batch[key] = torch.tensor(values)
    return batch


def get_dataset(
    dataset_name,
    tokenizer,
    wrap,
    mode,
    cache_dir,
    insert_eos=True,
    insert_special_tokens=True,
    block_size=1024,
    num_proc=None,
    streaming=False,
    revision: Optional[str] = None,
    min_length: int = 0,
    chunking: str = "none",
    dataset_config: Optional[dict] = None,
    tokenize=True,
    load_from_cache=False,
):
    # fast fix for linux/macos
    if num_proc is None:
        try:
            num_proc = max(1, len(os.sched_getaffinity(0)) - 1)
        except AttributeError:
            num_proc = max(1, os.cpu_count() - 1)
    else:
        num_proc = max(1, num_proc)

    chunking_mode = (chunking or "none").lower()
    load_from_cache = bool(load_from_cache or LOAD_FROM_CACHE)
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
    processing_streaming = streaming
    cached_synth_builder = get_cached_synthetic_builder(
        dataset_name, dataset_config, tokenize=tokenize
    )
    if cached_synth_builder is not None:
        filename = cached_synth_builder.cache_filename(filename)
    _path = os.path.join(cache_dir, filename)
    if cached_synth_builder is not None:
        cached_synth_builder.maybe_remove_existing_cache(_path)

    use_cached_streaming = (
        cached_synth_builder is not None
        and cached_synth_builder.should_use_cached_streaming(streaming=streaming)
    )
    if use_cached_streaming:
        if utils.fsspec_exists(_path):
            LOGGER.info(
                "Loading %s cached tokenized Arrow split from: %s",
                dataset_name.upper(),
                _path,
            )
            metadata_path = cached_synth_builder.cache_metadata_path(_path)
            if os.path.exists(metadata_path):
                LOGGER.info(
                    "%s cache metadata: %s",
                    dataset_name.upper(),
                    metadata_path,
                )
            # Arrow datasets loaded from disk are memory-mapped by HF Datasets.
            return _with_torch_collatable_format(datasets.load_from_disk(_path))

    can_reuse_cache = (
        cached_synth_builder is None or cached_synth_builder.should_reuse_cache()
    )
    if utils.fsspec_exists(_path) and load_from_cache and can_reuse_cache:
        LOGGER.info("Loading data from: %s", _path)
        return _with_torch_collatable_format(datasets.load_from_disk(_path))
    LOGGER.info("Generating new data at: %s", _path)
    LOGGER.info("streaming=%s", streaming)

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        block_size *= 2

    cached_synthetic_streaming_needs_build = False
    if dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext",
            name="wikitext-103-raw-v1",
            cache_dir=cache_dir,
            revision=revision,
        )
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "wikitext2-v1":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-v1", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "ptb":
        dataset = datasets.load_dataset(
            "ptb_text_only", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
    elif dataset_name == "text8":
        assert wrap
        assert revision is None
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "text8-crop":
        assert revision is None
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
            trust_remote_code=True,
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
            trust_remote_code=True,
        )
    elif dataset_name == "scientific_papers_arxiv":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "arxiv",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "scientific_papers_pubmed":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "pubmed",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news", cache_dir=cache_dir, streaming=streaming, revision=revision
        )
    elif dataset_name == "synthetic":
        assert streaming
        assert wrap
        dataset = generate_synthetic_dataset(
            train_dataset_size=100000,
            validation_dataset_size=1024,
            seq_len=32,
            vocab_size=256,
        )
    elif dataset_name == "star_graph":
        if streaming:
            raise ValueError(
                "star_graph dataset generation does not support streaming."
            )
        dataset = get_star_graph_dataset(dataset_config=dataset_config)
    elif dataset_name == "brevo":
        if cached_synth_builder is None:
            raise ValueError("BREVO dataset requires `data.dataset_config`.")
        if streaming and cached_synth_builder.cached_streaming:
            # Build in bounded shards instead of one monolithic in-memory split.
            dataset = None
            cached_synthetic_streaming_needs_build = True
            processing_streaming = False
        else:
            if streaming:
                # datasets==3.5.0 IterableDataset.from_generator() uses HF_DATASETS_CACHE.
                datasets.config.HF_DATASETS_CACHE = cache_dir
            dataset = get_brevo_dataset(
                dataset_config=cached_synth_builder.dataset_config,
                split=mode,
                streaming=streaming,
            )
    elif dataset_name == "prefix":
        if streaming:
            raise ValueError("prefix dataset generation does not support streaming.")
        dataset = generate_prefix_dataset()
    else:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
            revision=revision,
        )

    if dataset_name in ["lambada", "openwebtext-train", "openwebtext-valid"]:
        data = dataset
    elif dataset_name == "brevo":
        data = dataset
    else:
        data = dataset[mode]
        if dataset_name == "synthetic":
            return data

    if dataset_name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif dataset_name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif dataset_name == "ptb":
        detokenizer = ptb_detokenizer
    elif dataset_name == "lambada":
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith("scientific_papers"):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    EOS = tokenizer.eos_token_id
    BOS = tokenizer.bos_token_id
    PAD = tokenizer.pad_token_id

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    use_chunking = chunking_mode != "none"
    if use_chunking:
        if chunking_mode == "double_newline":
            delimiter_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
        else:
            delimiter_tokens = []
        if not delimiter_tokens:
            raise ValueError(
                "Tokenizer did not produce any tokens for the specified chunking delimiter."
            )
    else:
        delimiter_tokens = []

    def preprocess_and_tokenize(example):
        if "prefixes" in example and "completions" in example:

            def _as_int_token_ids(value):
                if isinstance(value, str):
                    stripped = value.strip()
                    if not stripped:
                        return []
                    return [int(tok) for tok in stripped.split()]
                if isinstance(value, (list, tuple)):
                    return [int(tok) for tok in value]
                if hasattr(value, "tolist"):
                    converted = value.tolist()
                    if isinstance(converted, list):
                        return [int(tok) for tok in converted]
                raise TypeError(
                    f"Unsupported pretokenized sequence type: {type(value)!r}"
                )

            if tokenize:
                prefixes = tokenizer(
                    example["prefixes"],
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                completions = tokenizer(
                    example["completions"],
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
            else:
                prefixes = {"input_ids": example["prefixes"]}
                completions = {"input_ids": example["completions"]}
            input_ids = []
            attention_mask = []
            loss_mask = []
            accuracy_mask = []
            noise_mask = []
            for p_ids, c_ids in zip(prefixes["input_ids"], completions["input_ids"]):
                if tokenize:
                    p_ids = list(p_ids)
                    c_ids = list(c_ids)
                else:
                    p_ids = _as_int_token_ids(p_ids)
                    c_ids = _as_int_token_ids(c_ids)

                # BREVO tokenize=False uses native task IDs (already includes task markers),
                # so avoid injecting tokenizer BOS/EOS again.
                add_boundary_tokens = dataset_name in {"prefix", "star_graph"} or (
                    dataset_name == "brevo" and tokenize
                )
                if add_boundary_tokens and BOS is not None:
                    if not p_ids or p_ids[0] != BOS:
                        p_ids = [BOS] + p_ids
                if add_boundary_tokens and EOS is not None:
                    if not c_ids or c_ids[-1] != EOS:
                        c_ids = c_ids + [EOS]

                # Truncate before padding; keep at least one completion token when possible.
                if len(p_ids) >= block_size and len(c_ids) > 0:
                    p_ids = p_ids[: max(block_size - 1, 0)]
                else:
                    p_ids = p_ids[:block_size]
                c_ids = c_ids[: max(block_size - len(p_ids), 0)]

                seq_ids = p_ids + c_ids
                seq_attention_mask = [1] * len(seq_ids)
                seq_loss_mask = [0] * len(p_ids) + [1] * len(c_ids)
                seq_accuracy_mask = [0] * len(p_ids) + [1] * len(c_ids)
                seq_noise_mask = [0] * len(p_ids) + [1] * len(c_ids)

                pad_len = block_size - len(seq_ids)
                if pad_len > 0:
                    seq_ids += [PAD] * pad_len
                    seq_attention_mask += [0] * pad_len
                    seq_loss_mask += [0] * pad_len
                    seq_accuracy_mask += [0] * pad_len
                    seq_noise_mask += [0] * pad_len

                input_ids.append(seq_ids)
                attention_mask.append(seq_attention_mask)
                loss_mask.append(seq_loss_mask)
                accuracy_mask.append(seq_accuracy_mask)
                noise_mask.append(seq_noise_mask)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "accuracy_mask": accuracy_mask,
                "noise_mask": noise_mask,
            }

        if dataset_name == "ptb":
            text = example["sentence"]
        elif "scientific_papers" in dataset_name:
            text = example["article"]
        else:
            text = example["text"]
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)
        if use_chunking:
            return chunk_documents(
                tokenizer,
                text,
                max_length=block_size,
                delimiter_tokens=delimiter_tokens,
                add_special_tokens=insert_special_tokens,
            )
        if wrap:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            if insert_eos:
                tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
        else:
            tokens = tokenizer(
                text,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                add_special_tokens=insert_special_tokens,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        return tokens

    def _remove_raw_columns(ds):
        column_names = ds.column_names or []
        if dataset_name == "ptb" and "sentence" in column_names:
            return ds.remove_columns("sentence")
        if "scientific_papers" in dataset_name and {
            "article",
            "abstract",
            "section_names",
        }.issubset(set(column_names)):
            return ds.remove_columns(["article", "abstract", "section_names"])
        if dataset_name == "ag_news" and {"text", "label"}.issubset(set(column_names)):
            return ds.remove_columns(["text", "label"])
        if "text" in column_names:
            return ds.remove_columns("text")
        return ds

    if cached_synthetic_streaming_needs_build:
        if cached_synth_builder is None:
            raise RuntimeError(
                "Internal error: cached synthetic builder missing for shard build."
            )

        split_size = cached_synth_builder.split_size(mode)
        if split_size <= 0:
            raise ValueError(
                f"{dataset_name.upper()} split size must be > 0 for mode={mode}"
            )

        num_shards = cached_synth_builder.cached_streaming_num_shards
        if num_shards <= 0:
            raise ValueError(
                f"`cached_streaming_num_shards` must be > 0, got {num_shards}"
            )
        num_shards = min(num_shards, split_size)
        shard_size = (split_size + num_shards - 1) // num_shards

        LOGGER.info(
            "Building %s cached tokenized split in %d shard(s) of up to %d examples",
            dataset_name.upper(),
            num_shards,
            shard_size,
        )

        shard_root = os.path.join(
            cache_dir,
            f".{cached_synth_builder.dataset_name}_cached_build_{mode}_{cached_synth_builder.cache_key}",
        )
        if os.path.isdir(shard_root):
            shutil.rmtree(shard_root)
        os.makedirs(shard_root, exist_ok=True)

        shard_paths = []
        split_records = cached_synth_builder.iter_split_records(mode)
        remaining = split_size

        for shard_idx in range(num_shards):
            current_size = min(shard_size, remaining)
            remaining -= current_size
            LOGGER.info(
                "Generating %s %s shard %d/%d (%d examples)",
                dataset_name.upper(),
                mode,
                shard_idx + 1,
                num_shards,
                current_size,
            )
            prefixes = []
            completions = []
            for _ in range(current_size):
                record = next(split_records)
                prefixes.append(record["prefixes"])
                completions.append(record["completions"])

            shard_dataset = datasets.Dataset.from_dict(
                {"prefixes": prefixes, "completions": completions}
            )
            shard_tokenized = shard_dataset.map(
                preprocess_and_tokenize,
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache,
                desc=(
                    f"Tokenizing {dataset_name.upper()} {mode} shard "
                    f"{shard_idx + 1}/{num_shards}"
                ),
            )
            shard_tokenized = _remove_raw_columns(shard_tokenized)

            if (not wrap) and min_length > 0:

                def _has_min_length(example):
                    mask = example.get("attention_mask", None)
                    if mask is None:
                        return True
                    return sum(mask) >= min_length

                shard_tokenized = shard_tokenized.filter(
                    _has_min_length,
                    num_proc=num_proc,
                    load_from_cache_file=load_from_cache,
                    desc=(
                        f"Filtering {dataset_name.upper()} {mode} shard "
                        f"{shard_idx + 1}/{num_shards}"
                    ),
                )

            shard_path = os.path.join(shard_root, f"shard_{shard_idx:05d}")
            shard_tokenized.save_to_disk(shard_path)
            shard_paths.append(shard_path)

        merged = datasets.concatenate_datasets(
            [datasets.load_from_disk(path) for path in shard_paths]
        )
        merged.save_to_disk(_path)
        cached_synth_builder.write_cache_metadata(_path, mode)
        shutil.rmtree(shard_root, ignore_errors=True)

        LOGGER.info(
            "Loading %s cached tokenized Arrow split from: %s",
            dataset_name.upper(),
            _path,
        )
        return _with_torch_collatable_format(datasets.load_from_disk(_path))

    map_kwargs = {
        "batched": True,
    }
    if use_chunking:
        map_kwargs["remove_columns"] = ["text"]
    if not processing_streaming:
        map_kwargs.update(
            num_proc=num_proc, load_from_cache_file=load_from_cache, desc="Tokenizing"
        )
    tokenized_dataset = data.map(preprocess_and_tokenize, **map_kwargs)
    tokenized_dataset = _remove_raw_columns(tokenized_dataset)

    if (not wrap) and min_length > 0 and (not processing_streaming):

        def _has_min_length(example):
            mask = example.get("attention_mask", None)
            if mask is None:
                return True
            return sum(mask) >= min_length

        tokenized_dataset = tokenized_dataset.filter(
            _has_min_length,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache,
            desc="Filtering min length",
        )

    if not wrap:
        if not processing_streaming:
            tokenized_dataset.save_to_disk(_path)
            if cached_synth_builder is not None:
                cached_synth_builder.write_cache_metadata(_path, mode)
            if use_cached_streaming:
                LOGGER.info(
                    "Loading %s cached tokenized Arrow split from: %s",
                    dataset_name.upper(),
                    _path,
                )
                return _with_torch_collatable_format(datasets.load_from_disk(_path))
        return _with_torch_collatable_format(tokenized_dataset)

    group_texts = functools.partial(
        _group_texts,
        block_size=block_size,
        bos=BOS,
        eos=EOS,
        insert_special_tokens=insert_special_tokens,
    )
    if processing_streaming:
        chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
    else:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache,
            desc="Grouping",
        )
        chunked_dataset.save_to_disk(_path)
    return _with_torch_collatable_format(chunked_dataset)


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == "text8":
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == "ascii-char":
        tokenizer = AsciiCharTokenizer()
    elif config.data.tokenizer_name_or_path == "brevo-dummy":
        tokenizer = BrevoDummyTokenizer()
    elif config.data.tokenizer_name_or_path == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif config.data.tokenizer_name_or_path == "synthetic":
        tokenizer = SyntheticTokenizer(vocab_size=256)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path
        )
    if isinstance(
        tokenizer, (transformers.GPT2TokenizerFast, transformers.GPT2Tokenizer)
    ):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                f"Tokenizer must have a bos_token or cls_token: {tokenizer}"
            )
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                f"Tokenizer must have a eos_token or sep_token: {tokenizer}"
            )
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    add_mask_token = config.algo.get(
        "add_mask_token", config.data.get("add_mask_token", True)
    )
    if add_mask_token and getattr(tokenizer, "mask_token", None) is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    return tokenizer


def _configured_device_count(config):
    devices = config.trainer.devices
    if isinstance(devices, int):
        return max(1, devices)
    if isinstance(devices, str):
        if devices.isdigit():
            return max(1, int(devices))
        if "," in devices:
            return len([device for device in devices.split(",") if device.strip()])
        return max(1, torch.cuda.device_count())
    try:
        return max(1, len(devices))
    except TypeError:
        return max(1, torch.cuda.device_count())


def get_dataloaders(
    config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None
):
    num_devices = _configured_device_count(config)
    assert config.loader.global_batch_size == (
        config.loader.batch_size
        * config.trainer.num_nodes
        * num_devices
        * config.trainer.accumulate_grad_batches
    )
    if (
        config.loader.global_batch_size
        % (num_devices * config.trainer.accumulate_grad_batches)
        != 0
    ):
        raise ValueError(
            f"Train Batch Size {config.loader.batch_size} "
            f"not divisible by {num_devices} devices with accumulation "
            f"{config.trainer.accumulate_grad_batches}."
        )
    if config.loader.eval_global_batch_size % num_devices != 0:
        raise ValueError(
            f"Eval Batch Size {config.loader.eval_global_batch_size} "
            f"not divisible by {num_devices} devices."
        )
    default_chunking = config.data.get("chunking", "none")
    train_chunking = config.data.get("train_chunking", default_chunking)
    valid_chunking = config.data.get("valid_chunking", default_chunking)
    dataset_config = config.data.get("dataset_config", None)
    load_from_cache = config.data.get("load_from_cache", False)
    if skip_train:
        train_set = None
    else:
        train_min_length = config.data.get(
            "train_min_length", config.data.get("min_length", 0)
        )
        train_set = get_dataset(
            config.data.train,
            tokenizer,
            mode="train",
            wrap=config.data.wrap,
            insert_eos=config.data.insert_train_eos,
            insert_special_tokens=getattr(config.data, "insert_train_special", True),
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("train_revision", None),
            min_length=train_min_length,
            chunking=train_chunking,
            dataset_config=dataset_config,
            load_from_cache=load_from_cache,
            tokenize=not (
                config.data.train == "brevo"
                and config.data.tokenizer_name_or_path == "brevo-dummy"
            ),
        )

    if config.data.valid in ["text8", "lm1b", "ag_news"]:
        validation_split = "test"
    else:
        validation_split = "validation"
    if skip_valid:
        valid_set = None
    else:
        valid_min_length = config.data.get(
            "valid_min_length", config.data.get("min_length", 0)
        )
        valid_set = get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            insert_eos=config.data.insert_valid_eos,
            insert_special_tokens=getattr(config.data, "insert_valid_special", True),
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("valid_revision", None),
            min_length=valid_min_length,
            chunking=valid_chunking,
            dataset_config=dataset_config,
            load_from_cache=load_from_cache,
            tokenize=not (
                config.data.valid == "brevo"
                and config.data.tokenizer_name_or_path == "brevo-dummy"
            ),
        )

    if skip_train:
        train_loader = None
    else:
        shuffle_train = config.loader.get("shuffle_train", None)
        if shuffle_train is None:
            shuffle_train = not config.data.streaming
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=bool(shuffle_train),
            persistent_workers=config.loader.num_workers > 0,
            collate_fn=_collate_examples,
        )
        train_loader.tokenizer = tokenizer
    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            default_shuffle_valid = False
            generator = None
        else:
            default_shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        shuffle_valid = config.loader.get("shuffle_valid", None)
        if shuffle_valid is None:
            shuffle_valid = default_shuffle_valid
        else:
            shuffle_valid = bool(shuffle_valid)
            if not shuffle_valid:
                generator = None
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
            persistent_workers=config.loader.num_workers > 0,
            collate_fn=_collate_examples,
        )
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader
