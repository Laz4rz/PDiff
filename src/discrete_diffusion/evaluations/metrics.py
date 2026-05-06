from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union

import torch
import torchmetrics

LOG2 = math.log(2)

Value = Union[float, torch.Tensor]


class NLL(torchmetrics.aggregation.MeanMetric):
    def update(self, value: Value, weight: Value = 1.0) -> None:
        """Update state with data.

        Args:
          value: Either a float or tensor containing data.
            Additional tensor dimensions will be flattened
          weight: Either a float or tensor containing weights
            for calculating the average. Shape of weight should
            be able to broadcast with the shape of `value`.
            Default to `1.0` corresponding to simple harmonic
            average.
        """
        # broadcast weight to value shape
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        else:
            value = value.to(dtype=self.dtype, device=self.device)

        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        else:
            weight = weight.to(dtype=self.dtype, device=self.device)

        # Handle edge case where torch.compile infers scalar value but sees tensor inputs
        if value.ndim == 0 and weight.ndim > 0:
            weight = weight.squeeze()
        if value.ndim > 0:
            weight = torch.broadcast_to(weight, value.shape)

        if value.numel() == 0:
            return
        self.mean_value += value.sum()
        self.weight += weight.sum()


class BPD(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> torch.Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class Metrics:
    def __init__(self, *_args, **_kwargs) -> None:
        metrics = torchmetrics.MetricCollection(
            {"nll": NLL(), "bpd": BPD(), "ppl": Perplexity()}
        )
        metrics.set_dtype(torch.float64)
        self.train_nlls = metrics.clone(prefix="train/")
        self.train_acc_token = NLL()
        self.train_acc_token.set_dtype(torch.float64)
        self.train_acc_sample = NLL()
        self.train_acc_sample.set_dtype(torch.float64)
        self.train_aux = BPD()
        self.valid_nlls = metrics.clone(prefix="val/")
        self.valid_acc_token = NLL()
        self.valid_acc_token.set_dtype(torch.float64)
        self.valid_acc_sample = NLL()
        self.valid_acc_sample.set_dtype(torch.float64)
        self.valid_gen_acc_token = NLL()
        self.valid_gen_acc_token.set_dtype(torch.float64)
        self.valid_gen_acc_sample = NLL()
        self.valid_gen_acc_sample.set_dtype(torch.float64)
        self.valid_aux = BPD()
        # Keep sample entropy as a lightweight generative signal during training
        self.sample_entropy = torchmetrics.aggregation.MeanMetric()
        self.sample_entropy.set_dtype(torch.float64)

    @staticmethod
    def _target_device(*args, **kwargs) -> torch.device | None:
        device = kwargs.get("device", None)
        if device is not None:
            return torch.device(device)
        for arg in args:
            if isinstance(arg, torch.device):
                return arg
            if isinstance(arg, str):
                return torch.device(arg)
            if isinstance(arg, torch.Tensor):
                return arg.device
        return None

    @classmethod
    def _move_metric(cls, metric, *args, **kwargs):
        target_device = cls._target_device(*args, **kwargs)
        if target_device is not None and target_device.type == "mps":
            metric.set_dtype(torch.float32)
            kwargs = dict(kwargs)
            if kwargs.get("dtype") is torch.float64:
                kwargs["dtype"] = torch.float32
        return metric.to(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.sample_entropy = self._move_metric(self.sample_entropy, *args, **kwargs)
        self.train_nlls = self._move_metric(self.train_nlls, *args, **kwargs)
        self.train_acc_token = self._move_metric(self.train_acc_token, *args, **kwargs)
        self.train_acc_sample = self._move_metric(
            self.train_acc_sample, *args, **kwargs
        )
        self.train_aux = self._move_metric(self.train_aux, *args, **kwargs)
        self.valid_nlls = self._move_metric(self.valid_nlls, *args, **kwargs)
        self.valid_acc_token = self._move_metric(self.valid_acc_token, *args, **kwargs)
        self.valid_acc_sample = self._move_metric(
            self.valid_acc_sample, *args, **kwargs
        )
        self.valid_gen_acc_token = self._move_metric(
            self.valid_gen_acc_token, *args, **kwargs
        )
        self.valid_gen_acc_sample = self._move_metric(
            self.valid_gen_acc_sample, *args, **kwargs
        )
        self.valid_aux = self._move_metric(self.valid_aux, *args, **kwargs)

    def reset(self):
        self.sample_entropy.reset()
        self.train_nlls.reset()
        self.train_acc_token.reset()
        self.train_acc_sample.reset()
        self.train_aux.reset()
        self.valid_nlls.reset()
        self.valid_acc_token.reset()
        self.valid_acc_sample.reset()
        self.valid_gen_acc_token.reset()
        self.valid_gen_acc_sample.reset()
        self.valid_aux.reset()

    def update_train(
        self,
        nll,
        num_tokens,
        correct_tokens=None,
        num_accuracy_tokens=None,
        correct_samples=None,
        num_accuracy_samples=None,
    ):
        self.train_nlls.update(nll, num_tokens)
        if correct_tokens is not None and num_accuracy_tokens is not None:
            self.train_acc_token.update(correct_tokens, num_accuracy_tokens)
        if correct_samples is not None and num_accuracy_samples is not None:
            self.train_acc_sample.update(correct_samples, num_accuracy_samples)

    def update_valid(
        self,
        nll,
        num_tokens,
        correct_tokens=None,
        num_accuracy_tokens=None,
        correct_samples=None,
        num_accuracy_samples=None,
    ):
        self.valid_nlls.update(nll, num_tokens)
        if correct_tokens is not None and num_accuracy_tokens is not None:
            self.valid_acc_token.update(correct_tokens, num_accuracy_tokens)
        if correct_samples is not None and num_accuracy_samples is not None:
            self.valid_acc_sample.update(correct_samples, num_accuracy_samples)

    def update_valid_gen(
        self,
        correct_tokens=None,
        num_accuracy_tokens=None,
        correct_samples=None,
        num_accuracy_samples=None,
    ):
        if correct_tokens is not None and num_accuracy_tokens is not None:
            self.valid_gen_acc_token.update(correct_tokens, num_accuracy_tokens)
        if correct_samples is not None and num_accuracy_samples is not None:
            self.valid_gen_acc_sample.update(correct_samples, num_accuracy_samples)

    # Generative ppl logic removed; use standalone evaluation script instead.

    @torch.no_grad()
    def record_entropy(self, tokens):
        for sample in tokens:
            entropy = _token_entropy(sample)
            self._record_entropy_value(entropy)

    def _record_entropy_value(self, entropy: float) -> None:
        self.sample_entropy.update(entropy)


class NFEs(torchmetrics.aggregation.MeanMetric):
    """Average number of function evaluations per sample."""


class BD3Metrics(Metrics):
    """Extension of Metrics with BD3-specific variance tracking."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.block_size = getattr(config, "block_size", config.model.length)
        self.nfes = NFEs()
        self.gen_entropy = NLL()
        self.gen_nfes: List[float] = []
        self.gen_entropies: List[float] = []
        self.gen_lengths: List[int] = []

        self.sampling_eps = config.training.sampling_eps
        self.clip_search_delta = getattr(config.algo, "clip_search_delta", None)
        self.valid_vars: Dict[Tuple[float, float], List[torch.Tensor]] = {
            (self.sampling_eps, 1.0): []
        }
        if getattr(config.algo, "var_min", None):
            self.init_valid_vars()

    def init_valid_vars(self):
        eps = self.sampling_eps
        if self.block_size > 1:
            self.valid_vars = {(eps, 1): []}
            for width in self.config.algo.clip_search_widths:
                for i in torch.arange(
                    0, 1 - width + self.clip_search_delta, self.clip_search_delta
                ):
                    eps_min = torch.clamp(i, min=self.sampling_eps).item()
                    eps_max = torch.clamp(i + width, min=self.sampling_eps).item()
                    self.valid_vars[(eps_min, eps_max)] = []
        else:
            self.valid_vars = {(eps, 1): [], (1, 1): []}

    def update_train(self, nll_sum: torch.Tensor, num_tokens: torch.Tensor):
        self.train_nlls.update(nll_sum, num_tokens)

    def update_valid(self, nll_sum: torch.Tensor, num_tokens: torch.Tensor):
        self.valid_nlls.update(nll_sum, num_tokens)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.nfes = self._move_metric(self.nfes, *args, **kwargs)
        self.gen_entropy = self._move_metric(self.gen_entropy, *args, **kwargs)

    def reset(self):
        super().reset()
        self.gen_nfes, self.gen_entropies, self.gen_lengths = [], [], []
        self.nfes.reset()
        self.gen_entropy.reset()
        if getattr(self.config.algo, "var_min", None):
            self.init_valid_vars()

    @torch.no_grad()
    def record_entropy(self, tokens):
        for sample in tokens:
            entropy = _token_entropy(sample)
            self._record_entropy_value(entropy)

    def _record_entropy_value(self, entropy: float) -> None:
        self.sample_entropy.update(entropy)
        self.gen_entropies.append(entropy)
        self.gen_entropy.update(entropy, 1)


def _token_entropy(sample: torch.Tensor) -> float:
    _, counts = torch.unique(sample, return_counts=True, sorted=False)
    probs = counts.float() / counts.sum()
    return torch.special.entr(probs).sum().item()


__all__ = ["Metrics", "BD3Metrics"]
