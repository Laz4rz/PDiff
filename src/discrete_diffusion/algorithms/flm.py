"""Flow Language Model (FLM) and PSD FMLM implementations.

The implementation mirrors the FLM reference repository while fitting PDiff's
Hydra/Lightning structure. FLM trains a denoiser on continuous one-hot/noise
interpolants. FMLM implements the default PSD denoiser distillation path.
"""

from __future__ import annotations

import collections
from typing import Iterable

import hydra.utils
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr

from . import base as trainer_base
from .. import utils


def _stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def _resolve_num_steps(num_steps) -> int:
    if isinstance(num_steps, omegaconf.ListConfig):
        num_steps = list(num_steps)
    if isinstance(num_steps, (list, tuple)):
        if not num_steps:
            raise ValueError("sampling.steps must not be empty")
        num_steps = num_steps[0]
    return int(num_steps)


def _compute_alpha_exact(
    gamma: np.ndarray,
    K: int,
    n_gh: int = 100,
    sigma_floor: float = 1e-12,
) -> np.ndarray:
    gamma = np.asarray(gamma)
    sigma = np.maximum(1.0 - gamma, sigma_floor)
    m_c = gamma / sigma

    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    log_cdf = log_ndtr(z_nodes[None, :] + m_c[:, None])
    q_c = np.sum(w * np.exp((K - 1) * log_cdf), axis=-1)
    alpha = K / (K - 1.0) * (q_c - 1.0 / K)
    alpha += (gamma - 1.0) * 1e-10
    return np.clip(alpha, 0.0, 1.0)


def _build_luts(K: int, n_points: int = 10_000) -> tuple[CubicSpline, CubicSpline]:
    gamma_vals = np.linspace(0.0, 1.0, n_points)
    alpha_vals = _compute_alpha_exact(gamma_vals, K=K)

    lut_g2a = CubicSpline(gamma_vals, alpha_vals)

    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]
    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]
    lut_a2g = CubicSpline(unique_alpha, unique_gamma)
    return lut_a2g, lut_g2a


def _spline_eval_torch(
    values: torch.Tensor, lut: CubicSpline, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    out_dtype = values.dtype if dtype is None else dtype
    out = np.clip(lut(values.detach().cpu().numpy()), 0.0, 1.0)
    return torch.from_numpy(out).to(values.device, dtype=out_dtype)


class FLMBase(trainer_base.TrainerBase):
    """Shared continuous-state helpers for FLM/FMLM."""

    def __init__(self, config, tokenizer):
        self._use_flm_dit_for_default_dit(config)
        super().__init__(config, tokenizer)
        self.t_min = float(config.algo.t_min)
        self.t_max = float(config.algo.t_max)
        self.lut_a2g, self.lut_g2a = _build_luts(K=self.vocab_size)
        self._is_resuming = bool(
            omegaconf.OmegaConf.select(
                config, "checkpointing.resume_from_ckpt", default=False
            )
            and omegaconf.OmegaConf.select(
                config, "checkpointing.resume_ckpt_path", default=None
            )
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )

    @staticmethod
    def _use_flm_dit_for_default_dit(config):
        target = omegaconf.OmegaConf.select(config, "model._target_", default=None)
        if target == "discrete_diffusion.models.dit.DIT":
            config.model._target_ = "discrete_diffusion.models.flm_dit.FLMDIT"

    def _validate_configuration(self):
        if self.config.algo.parameterization != "mean":
            raise ValueError("FLM/FMLM require algo.parameterization=mean")
        if not self.config.algo.time_conditioning:
            raise ValueError("FLM/FMLM require time conditioning")
        if self.config.algo.T != 0:
            raise ValueError("FLM/FMLM support only continuous-time T=0")

    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        valid_tokens = batch.get("loss_mask", batch["attention_mask"])
        losses = self._loss(
            batch["input_ids"],
            valid_tokens,
            current_accumulation_step,
            train_mode=True,
            xT=batch.get("xT"),
            given_t=batch.get("given_t"),
            not_sampling_t=bool(
                getattr(self.config.training, "not_sampling_t", False)
            ),
        )
        self.metrics.update_train(losses.nlls, losses.num_tokens)
        self.log(
            name="trainer/loss",
            value=losses.loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            name="train/loss",
            value=losses.loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return losses.loss

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value: float = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _loss(
        self,
        x0,
        valid_tokens,
        current_accumulation_step=None,
        train_mode=False,
        xT=None,
        given_t=None,
        not_sampling_t=False,
    ):
        input_tokens, output_tokens, valid_tokens = self._process_model_input(
            x0, valid_tokens
        )
        loss = self.loss(
            input_tokens,
            output_tokens,
            current_accumulation_step,
            train_mode,
            xT=xT,
            given_t=given_t,
            not_sampling_t=not_sampling_t,
        )
        assert loss.ndim == 2
        if self.ignore_bos:
            loss = loss.clone()
            valid_tokens = valid_tokens.clone()
            loss[:, 0] = 0
            valid_tokens[:, 0] = 0

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens
        return trainer_base.Loss(loss=token_nll, nlls=nlls, num_tokens=num_tokens)

    def loss(
        self,
        x0,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        xT=None,
        given_t=None,
        not_sampling_t=False,
    ):
        raise NotImplementedError

    def nll(self, input_tokens, current_accumulation_step=None, train_mode=False):
        return self.loss(
            input_tokens,
            None,
            current_accumulation_step=current_accumulation_step,
            train_mode=train_mode,
        )

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            eps_t = (eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            eps_t = eps_t[perm]
        t = (t_max - t_min) * eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t = t[:batch_dim]
        return t

    def _tau_to_t(self, tau: torch.Tensor) -> torch.Tensor:
        return _spline_eval_torch(tau, self.lut_a2g)

    def _t_to_tau(self, t: torch.Tensor) -> torch.Tensor:
        return _spline_eval_torch(t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        t = t.unsqueeze(-1).unsqueeze(-1)
        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        self._is_resuming = True
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = self._filter_checkpoint_state_dict(
                checkpoint["state_dict"]
            )
        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = collections.OrderedDict(
                (k, v)
                for k, v in checkpoint["state_dict"].items()
                if not k.startswith("teacher")
            )
        super().on_save_checkpoint(checkpoint)

    def _filter_checkpoint_state_dict(self, state_dict):
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("teacher"):
                continue
            new_state_dict[key.replace("._orig_mod.", ".")] = value
        return new_state_dict

    def forward(self, xt, sigma, sigma_prime=None, use_jvp_attn=False, **kwargs):
        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(
                xt, sigma, sigma_prime=sigma_prime, use_jvp_attn=use_jvp_attn, **kwargs
            )
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    def forward_no_softmax(self, xt, tau, tau_prime=None, **kwargs):
        tau = self._process_sigma(tau)
        if tau_prime is not None:
            tau_prime = self._process_sigma(tau_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            return self.backbone(xt, tau, sigma_prime=tau_prime, **kwargs)

    def _extract_ema_state_dict(self, model, checkpoint):
        ema_state = checkpoint.get("ema", None)
        if not ema_state:
            return {
                k.replace("backbone.", "").replace("._orig_mod.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("backbone.")
            }

        new_state_dict = collections.OrderedDict()
        shadow_params = ema_state["shadow_params"]
        param_names = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        for name, value in zip(param_names, shadow_params):
            new_state_dict[name] = value

        named_parameters = {name for name, _ in model.named_parameters()}
        for key, value in checkpoint["state_dict"].items():
            clean_key = key.replace("backbone.", "").replace("._orig_mod.", "")
            if clean_key not in new_state_dict and clean_key in named_parameters:
                new_state_dict[clean_key] = value
        return new_state_dict

    def _load_teacher_model(self, path, use_plain_config=True):
        if path is None or path == "":
            return None

        if use_plain_config:
            saved = (
                bool(self.config.algo.double_temb),
                bool(self.config.algo.learnable_loss_weighting),
            )
            self.config.algo.double_temb = False
            self.config.algo.learnable_loss_weighting = False

        target = self.config.model._target_
        instantiate_config = omegaconf.OmegaConf.create({"_target_": target})
        model = hydra.utils.instantiate(
            instantiate_config, self.config, self.vocab_size, _recursive_=False
        )

        if use_plain_config:
            self.config.algo.double_temb = saved[0]
            self.config.algo.learnable_loss_weighting = saved[1]

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _copy_teacher_weights_to_student(
        self, teacher_items: Iterable[tuple[str, torch.Tensor]]
    ):
        with torch.no_grad():
            student_dict = self.backbone.state_dict()
            for name, param in teacher_items:
                if name in student_dict and student_dict[name].shape == param.shape:
                    student_dict[name].copy_(param)
            sigma_map_prime = getattr(self.backbone, "sigma_map_prime", None)
            if sigma_map_prime is not None:
                for name, param in sigma_map_prime.named_parameters():
                    if "mlp.2" in name:
                        param.zero_()


class FLM(FLMBase):
    """Flow Language Model."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def loss(
        self,
        x0,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        xT=None,
        given_t=None,
        not_sampling_t=False,
    ):
        del output_tokens, train_mode, xT, given_t, not_sampling_t
        batch_size = x0.shape[0]
        tau_t = self._sample_t_interval(
            batch_size, current_accumulation_step, t_min=self.t_min, t_max=self.t_max
        )
        t = self._tau_to_t(tau_t)
        x_t, target_data = self.corrupt_continuous(x0, t)
        log_probs = self.forward(x_t, tau_t)
        loss = -(target_data * log_probs).sum(dim=-1)
        self.log("loss", loss.mean(), prog_bar=True)
        if bool(getattr(self.config.algo, "learnable_loss_weighting", False)):
            loss_weight = self.backbone.learnable_loss_weighting(tau_t).unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
            self.log("loss_weighted", loss.mean(), prog_bar=True)
        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, **kwargs):
        del eps, kwargs
        if num_steps is None:
            num_steps = self.config.sampling.steps
        num_steps = _resolve_num_steps(num_steps)
        batch_size = num_samples
        vocab_size = self.vocab_size
        seq_len = self.num_tokens
        device = self.device

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn(
            (num_samples, seq_len, vocab_size), device=device, dtype=self.dtype
        )

        for i in range(num_steps):
            tau_curr = tau_vals[i]
            tau_next = tau_vals[i + 1]
            tau_in = tau_curr.expand(batch_size)
            t_in = self._tau_to_t(tau_in)
            dt = self._tau_to_t(tau_next.expand(batch_size)) - t_in
            x_1_pred = self.forward(z, tau_in).exp()
            if i == num_steps - 1:
                z = x_1_pred
                break
            v = (x_1_pred - z) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
            z = z + dt.view(-1, 1, 1) * v
        return z.argmax(dim=-1)


class FMLM(FLMBase):
    """Flow Map Language Model denoiser with PSD distillation."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.teacher_model = None

    def setup(self, stage: str):
        if self.teacher_model is None:
            self._initialize_teacher()
        if stage == "fit" and not self._is_resuming:
            self._initialize_student_from_teacher()

    def _initialize_teacher(self):
        self.teacher_model = self._load_teacher_model(
            self.config.algo.teacher_path, use_plain_config=True
        )

    def _initialize_student_from_teacher(self):
        if self.teacher_model is None:
            return
        if not bool(getattr(self.config.algo, "initialize_student_from_teacher", True)):
            return
        self._copy_teacher_weights_to_student(self.teacher_model.state_dict().items())

    def _validate_configuration(self):
        super()._validate_configuration()
        if not bool(self.config.algo.double_temb):
            raise ValueError("FMLM denoiser requires algo.double_temb=true")
        if not isinstance(self.config.algo.diagonal_fraction, float):
            raise ValueError("diagonal_fraction must be a float")
        if not 0 <= self.config.algo.diagonal_fraction <= 1:
            raise ValueError("diagonal_fraction must be between 0 and 1")
        if self.config.algo.distillation_method not in {"PSD", "LSD", "ESD"}:
            raise ValueError("FMLM distillation_method must be PSD, LSD, or ESD")

    def forward_with_ema(self, *args, **kwargs):
        if self.ema is None:
            raise ValueError("EMA must be available")
        self.ema.store(self._get_parameters())
        self.ema.copy_to(self._get_parameters())
        try:
            with torch.no_grad():
                self.backbone.eval()
                return self.forward(*args, **kwargs)
        finally:
            self.ema.restore(self._get_parameters())
            self.backbone.train()

    def teacher_forward(self, xt, tau=None, d=None, use_jvp_attn=False):
        del d, use_jvp_attn
        if self.teacher_model is None:
            raise ValueError("teacher_forward requires a loaded teacher model")
        sigma = tau.unsqueeze(-1) if tau.ndim == 1 else tau
        sigma = self._process_sigma(sigma)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.teacher_model(xt, sigma)
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    def _sample_multi_t_interval(
        self, n, accum_step, num_per_sample, t_min=None, t_max=None
    ):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max

        if accum_step is not None:
            batch_dim = n
            total_n = self.config.loader.global_batch_size * num_per_sample
        else:
            batch_dim = n
            total_n = n * num_per_sample

        eps_t = torch.rand(total_n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(total_n, device=self.device) / total_n
            eps_t = (eps_t / total_n + offset) % 1
            perm = torch.randperm(total_n, device=self.device)
            eps_t = eps_t[perm]

        t_all = (t_max - t_min) * eps_t + t_min
        if accum_step is not None:
            t_all = t_all.view(-1, num_per_sample)
            t_all = t_all.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t_all = t_all.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t_all = t_all.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t_all = t_all[:batch_dim]
        else:
            t_all = t_all.view(batch_dim, num_per_sample)

        t_sorted, _ = torch.sort(t_all, dim=-1)
        return [t_sorted[:, i] for i in range(num_per_sample)]

    def _get_split_indices(self, n, accum_step, ratios):
        if sum(ratios) >= 0.999:
            raise ValueError("Sum of split ratios must leave one category")

        batch_dim = n
        if accum_step is not None:
            n_total = self.config.loader.global_batch_size
            num_nodes = self.trainer.num_nodes
            num_devices = self.trainer.num_devices
            accum_batches = self.trainer.accumulate_grad_batches
            node_rank = self.trainer.node_rank
            local_rank = self.trainer.local_rank
            current_accum = accum_step
        else:
            n_total = n
            num_nodes = num_devices = accum_batches = 1
            node_rank = local_rank = current_accum = 0

        total_chunks = num_nodes * num_devices * accum_batches
        chunk_idx = (
            node_rank * num_devices * accum_batches
            + local_rank * accum_batches
            + current_accum
        )

        num_categories = len(ratios) + 1
        global_counts = [0] * num_categories
        net_ratio, prev_num = 0.0, 0
        for i, ratio in enumerate(ratios):
            net_ratio += ratio
            num_a = int(n_total * net_ratio)
            global_counts[i] = num_a - prev_num
            prev_num = num_a
        global_counts[-1] = n_total - prev_num

        local_counts = []
        for count in global_counts:
            base, rem = divmod(count, total_chunks)
            local_counts.append(base + (1 if chunk_idx < rem else 0))

        local_size = sum(local_counts)
        local_assignments = torch.empty(
            local_size, device=self.device, dtype=torch.long
        )
        offset = 0
        for i, count in enumerate(local_counts):
            local_assignments[offset : offset + count] = i
            offset += count

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.global_step * total_chunks + chunk_idx)
        perm = torch.randperm(local_size, device=self.device, generator=generator)
        local_assignments = local_assignments[perm][:batch_dim]
        return [
            (local_assignments == i).nonzero(as_tuple=True)[0]
            for i in range(num_categories)
        ]

    def loss(
        self,
        x1,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        xT=None,
        given_t=None,
        not_sampling_t=False,
    ):
        del output_tokens, train_mode, xT, given_t, not_sampling_t
        if self.config.algo.distillation_method != "PSD":
            raise NotImplementedError(
                "Only PSD FMLM distillation is implemented in PDiff's first FLM pass."
            )

        batch_size, seq_len = x1.shape[0], x1.shape[1]
        tau_diag = self._sample_t_interval(
            batch_size,
            current_accumulation_step,
            t_min=self.t_min,
            t_max=self.t_max,
        )
        if self.config.algo.offdiagonal_sampling == "uniform_st":
            tau_s_offdiag, tau_t_offdiag = self._sample_multi_t_interval(
                batch_size,
                current_accumulation_step,
                2,
                t_min=self.t_min,
                t_max=self.t_max,
            )
        else:
            tau_d_offdiag = self._sample_t_interval(
                batch_size,
                current_accumulation_step,
                t_min=self.t_min,
                t_max=self.t_max,
            )
            tau_s_offdiag = self._sample_t_interval(
                batch_size,
                current_accumulation_step,
                t_min=self.t_min,
                t_max=self.t_max,
            )
            tau_s_offdiag = tau_s_offdiag * (1 - tau_d_offdiag)
            tau_t_offdiag = tau_s_offdiag + tau_d_offdiag

        boundary_ratio = (1.0 - self.config.algo.diagonal_fraction) * (
            1.0 / self.config.algo.boundary_prob
        )
        idx_diag, idx_offdiag_bndry, idx_offdiag = self._get_split_indices(
            batch_size,
            current_accumulation_step,
            ratios=(self.config.algo.diagonal_fraction, boundary_ratio),
        )

        tau_s = torch.zeros(batch_size, device=self.device)
        tau_t = torch.zeros(batch_size, device=self.device)
        tau_s[idx_diag] = tau_diag[idx_diag]
        tau_s[idx_offdiag_bndry] = 0.0
        tau_s[idx_offdiag] = tau_s_offdiag[idx_offdiag]
        tau_t[idx_diag] = tau_diag[idx_diag]
        tau_t[idx_offdiag_bndry] = 1.0
        tau_t[idx_offdiag] = tau_t_offdiag[idx_offdiag]
        tau_s = torch.clamp(tau_s, 0.0, 1.0)
        tau_t = torch.clamp(tau_t, 0.0, 1.0)

        idx_offdiag = torch.cat([idx_offdiag, idx_offdiag_bndry])
        has_diag = idx_diag.numel() > 0
        has_offdiag = idx_offdiag.numel() > 0

        set_midpoint = getattr(self.config.algo, "set_midpoint", "midpoint")
        if set_midpoint == "midpoint":
            tau_u = 0.5 * (tau_s + tau_t)
        else:
            tau_u = tau_s + torch.rand_like(tau_s) * (tau_t - tau_s)

        s = self._tau_to_t(tau_s)
        u = self._tau_to_t(tau_u)
        t = self._tau_to_t(tau_t)
        x_s, target_data = self.corrupt_continuous(x1, s)

        if self.teacher_model is None or not has_diag:
            on_diagonal_target = target_data[idx_diag]
        else:
            on_diagonal_target = self.teacher_forward(
                x_s[idx_diag], tau_s[idx_diag]
            ).exp()
        on_diagonal_target = _stopgrad(on_diagonal_target)

        log_D_st = self.forward(x_s, tau_s, tau_t)
        target_forward = (
            self.forward_with_ema
            if bool(getattr(self.config.algo, "use_ema_for_psd_target", False))
            else self.forward
        )

        if has_offdiag:
            with torch.no_grad():
                x_s_od = x_s[idx_offdiag]
                s_od = s[idx_offdiag].view(-1, 1, 1)
                u_od = u[idx_offdiag].view(-1, 1, 1)
                t_od = t[idx_offdiag].view(-1, 1, 1)
                tau_s_od = tau_s[idx_offdiag]
                tau_u_od = tau_u[idx_offdiag]
                tau_t_od = tau_t[idx_offdiag]
                D_su_offdiag = target_forward(x_s_od, tau_s_od, tau_u_od).exp()
                X_su = ((1 - u_od) / (1 - s_od + 1e-8)) * x_s_od + (
                    (u_od - s_od) / (1 - s_od + 1e-8)
                ) * D_su_offdiag
                D_ut_offdiag = target_forward(X_su, tau_u_od, tau_t_od).exp()
                lambda_sut = (
                    (1 - t_od) * (u_od - s_od) / ((1 - u_od) * (t_od - s_od) + 1e-8)
                )
                offdiag_target = _stopgrad(
                    lambda_sut * D_su_offdiag + (1 - lambda_sut) * D_ut_offdiag
                )
            if not bool(self.config.algo.use_mse_loss_psd):
                offdiag_loss = -(offdiag_target * log_D_st[idx_offdiag]).sum(dim=-1)
            else:
                offdiag_loss = F.mse_loss(
                    log_D_st[idx_offdiag].exp(), offdiag_target, reduction="none"
                ).sum(dim=-1)
            if bool(self.config.algo.rescale_offdiag_loss_psd):
                offdiag_loss = offdiag_loss * ((t_od - s_od) / (1 - s_od + 1e-8)).view(
                    -1, 1
                ).pow(2)
        else:
            offdiag_loss = x_s.new_empty((0, seq_len))

        if has_diag:
            if not bool(self.config.algo.use_mse_loss_psd):
                diag_loss = -(on_diagonal_target * log_D_st[idx_diag]).sum(dim=-1)
            else:
                diag_loss = F.mse_loss(
                    log_D_st[idx_diag].exp(), on_diagonal_target, reduction="none"
                ).sum(dim=-1)
        else:
            diag_loss = x_s.new_empty((0, seq_len))

        loss = torch.zeros(batch_size, seq_len, device=self.device)
        if has_diag:
            loss[idx_diag] = diag_loss
            diag_loss_to_log = diag_loss.mean()
        else:
            diag_loss_to_log = loss.new_tensor(0.0)
        self.log("diag_loss", diag_loss_to_log, prog_bar=True, sync_dist=True)

        if has_offdiag:
            loss[idx_offdiag] = offdiag_loss
            offdiag_loss_to_log = offdiag_loss.mean()
        else:
            offdiag_loss_to_log = loss.new_tensor(0.0)
        self.log("offdiag_loss", offdiag_loss_to_log, prog_bar=True, sync_dist=True)
        self.log("loss", loss.mean(), prog_bar=True, sync_dist=True)

        if bool(getattr(self.config.algo, "learnable_loss_weighting", False)):
            loss_weight = self.backbone.learnable_loss_weighting(tau_s, tau_t)
            loss = torch.exp(-loss_weight.unsqueeze(-1)) * loss + loss_weight.unsqueeze(
                -1
            )
            self.log("loss_weighted", loss.mean(), prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5, **kwargs):
        del eps, kwargs
        if num_steps is None:
            num_steps = self.config.sampling.steps
        num_steps = _resolve_num_steps(num_steps)
        gamma = float(getattr(self.config.sampling, "gamma", 0.0))

        batch_size = num_samples
        vocab_size = self.vocab_size
        seq_len = self.num_tokens
        device = self.device
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn(
            (num_samples, seq_len, vocab_size), device=device, dtype=self.dtype
        )

        for i in range(num_steps):
            tau_curr = tau_vals[i]
            tau_next = tau_vals[i + 1]
            t_curr = self._tau_to_t(tau_curr.expand(batch_size))
            t_next = self._tau_to_t(tau_next.expand(batch_size))
            sigma_target = 1.0 - t_next
            sigma_tilde = sigma_target * torch.sqrt(
                torch.tensor(1.0 - gamma**2, device=device, dtype=t_next.dtype)
            )
            t_tilde = 1.0 - sigma_tilde
            tau_tilde = self._t_to_tau(t_tilde)

            D_st_pred = self.forward(z, tau_curr.expand(batch_size), tau_tilde).exp()
            if i == num_steps - 1:
                z = D_st_pred
                break

            weight_z = (1.0 - t_tilde.view(-1, 1, 1)) / (1.0 - t_curr.view(-1, 1, 1))
            weight_D = (t_tilde.view(-1, 1, 1) - t_curr.view(-1, 1, 1)) / (
                1.0 - t_curr.view(-1, 1, 1)
            )
            z_tilde = weight_z * z + weight_D * D_st_pred
            if gamma > 0:
                noise_std = gamma * sigma_target.view(-1, 1, 1)
                mean_adjustment = sigma_tilde.view(-1, 1, 1) - sigma_target.view(
                    -1, 1, 1
                )
                z = (
                    z_tilde
                    + mean_adjustment * D_st_pred
                    + noise_std * torch.randn_like(z)
                )
            else:
                z = z_tilde
        return z.argmax(dim=-1)


class UnsupportedFMLMVariant(FLMBase):
    """Placeholder for FLM source variants outside the first PDiff pass."""

    def __init__(self, config, tokenizer):
        variant = getattr(config.algo, "name", self.__class__.__name__)
        raise NotImplementedError(
            f"{variant} is not implemented in this PDiff FLM pass. "
            "Only FLM pretraining and default PSD FMLM denoiser distillation "
            "are supported."
        )


class FMLM_TwoModel(UnsupportedFMLMVariant):
    pass


class FMLM_TwoStage(UnsupportedFMLMVariant):
    pass


__all__ = [
    "FLM",
    "FMLM",
    "FLMBase",
    "UnsupportedFMLMVariant",
    "FMLM_TwoModel",
    "FMLM_TwoStage",
]
