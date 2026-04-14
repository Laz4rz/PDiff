"""GIDD algorithm implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf

from . import base as trainer_base
from ..noise_schedules import LogLinear
from ..noise_schedules import HybridDiffusion, sample_t as sample_t_hybrid
from ..noise_schedules.gidd_constant_pi import (
    GiddLinearNoise,
    sample_t as sample_t_constant_pi,
)


class GiddLoss(nn.Module):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.vocab_size = len(tokenizer)

        try:
            self.loss_weighting = config.loss.loss_weighting
            self.min_loss_weight = config.loss.min_loss_weight
            self.max_loss_weight = config.loss.max_loss_weight
        except Exception:
            self.loss_weighting = getattr(config.algo, "loss_weighting", "dynamic")
            self.min_loss_weight = float(getattr(config.algo, "min_loss_weight", 0.0))
            self.max_loss_weight = float(getattr(config.algo, "max_loss_weight", 2.0))
        assert self.max_loss_weight > 0, "max_loss_weight must be positive"

        self.mask_id = tokenizer.mask_token_id

    def get_weights(self, t: torch.Tensor, z_t: torch.Tensor, input_ids: torch.Tensor):
        orig_dtype = t.dtype
        t = t.unsqueeze(-1).to(torch.float64)
        t1m = 1 - t

        gamma = self.noise_schedule.log_gamma.exp()
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)
        B = self.noise_schedule.log_B.exp()

        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        c_t_prime = (gamma / 2) * (1 - 2 * t) / (t * t1m) * c_t

        is_mask = (z_t == self.mask_id).to(t.dtype)
        is_x = (z_t == input_ids).to(t.dtype)

        alpha_ratio = -1 / (1 - t) - c_t_prime / (1 + c_t)
        N = self.vocab_size - 1
        weight_on_x = (c_t + (1 - t) * c_t_prime) / N / ((1 - t) * (1 - t + c_t / N))
        weight_on_u = (c_t + (1 - t) * c_t_prime) / ((1 - t) * c_t)
        weight_on_m = 1 / ((1 - t) * t)

        elbo_weights = (
            is_x * weight_on_x
            + is_mask * weight_on_m
            + (1 - is_x - is_mask) * weight_on_u
        )

        loss_weights = elbo_weights.clone()
        if self.loss_weighting == "clip":
            loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "dynamic":
            log_snr_like = torch.sigmoid(-t).clip(-20, 20)
            x_scale = B / self.vocab_size * torch.exp(gamma / 2 * log_snr_like)
            loss_weights = (1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale
            loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "raw":
            loss_weights = elbo_weights

        return (
            alpha_ratio.to(orig_dtype),
            elbo_weights.to(orig_dtype),
            loss_weights.to(orig_dtype),
        )

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        reduction: str = "tokenmean",
    ):
        dtype = logits.dtype
        _, elbo_weights, ws = self.get_weights(t, z_t, input_ids)

        logits[..., self.mask_id] = torch.finfo(dtype).min

        x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
        x_hat = logits.softmax(-1).to(dtype)
        log_q_t = self.noise_schedule.probs_at_t(x, t).log_().clip_(min=-1e6)
        log_p_t = self.noise_schedule.probs_at_t(x_hat, t).log_().clip_(min=-1e6)

        kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

        log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_ratio = log_q_zt - log_p_zt

        is_loss = log_ratio.exp() - log_ratio - 1
        elbo = elbo_weights * (kl_loss + is_loss)
        loss = ws * (kl_loss + is_loss)

        eps = torch.finfo(loss.dtype).eps
        denom_ws = (ws * attention_mask).sum().clamp_min(eps)
        denom_attn = attention_mask.sum().clamp_min(eps)
        metrics = {
            "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
            "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
            "elbo": (elbo.detach() * attention_mask).sum()
            / denom_attn,
            "loss_weight_mean": (ws.detach() * attention_mask).sum() / denom_attn,
            "loss_weight_min": ws.detach().min(),
            "loss_weight_max": ws.detach().max(),
            "raw_weight_mean": (elbo_weights.detach() * attention_mask).sum()
            / denom_attn,
            "raw_weight_min": elbo_weights.detach().min(),
            "raw_weight_max": elbo_weights.detach().max(),
        }

        if reduction == "tokenmean":
            num_tokens = attention_mask.numel()
            loss = loss.sum() / num_tokens

        return loss, elbo, metrics


class GiddLossConstantPi(nn.Module):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.vocab_size = len(tokenizer)

        try:
            self.loss_weighting = config.loss.loss_weighting
            self.min_loss_weight = config.loss.min_loss_weight
            self.max_loss_weight = config.loss.max_loss_weight
        except Exception:
            self.loss_weighting = getattr(config.algo, "loss_weighting", "dynamic")
            self.min_loss_weight = float(getattr(config.algo, "min_loss_weight", 0.0))
            self.max_loss_weight = float(getattr(config.algo, "max_loss_weight", 2.0))
        assert self.max_loss_weight > 0, "max_loss_weight must be positive"

        self.mask_id = tokenizer.mask_token_id
        self.supports_mask = bool((self.noise_schedule.pi[self.mask_id] > 0).item())

    def _drop_mask_dim(self, x: torch.Tensor) -> torch.Tensor:
        mid = self.mask_id
        return torch.cat([x[..., :mid], x[..., mid + 1 :]], dim=-1)

    def _remap_no_mask(self, idx: torch.Tensor) -> torch.Tensor:
        mid = self.mask_id
        return idx - (idx > mid).to(idx.dtype)

    def get_weights(self, t: torch.Tensor, z_t: torch.Tensor, input_ids: torch.Tensor):
        orig_dtype = t.dtype
        t = t.unsqueeze(-1).to(torch.float64)

        alpha_ratio = 1 / (1 - t)

        is_x = (z_t == input_ids).to(t.dtype)

        pi_values = self.noise_schedule.gather_pi(input_ids)

        weight_on_not_x = 1 / t
        weight_on_x = pi_values / (1 - t + t * pi_values)

        elbo_weights = (is_x * weight_on_x + (1 - is_x) * weight_on_not_x) * alpha_ratio

        loss_weights = elbo_weights.clone()
        if self.loss_weighting == "clip":
            loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "dynamic":
            raise NotImplementedError("Unsure about it")
            log_snr_like = torch.sigmoid(-t).clip(-20, 20)
            x_scale = pi_values * torch.exp(log_snr_like)
            loss_weights = (1 - is_x) * ((1 - is_x) + 2 * is_x) + is_x * x_scale
            loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "raw":
            loss_weights = elbo_weights

        return (
            alpha_ratio.to(orig_dtype),
            elbo_weights.to(orig_dtype),
            loss_weights.to(orig_dtype),
        )

    def forward(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        reduction: str = "tokenmean",
    ):
        dtype = logits.dtype
        _, elbo_weights, ws = self.get_weights(t, z_t, input_ids)

        logits[..., self.mask_id] = torch.finfo(dtype).min

        if self.supports_mask:
            x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
            x_hat = logits.softmax(-1).to(dtype)
            log_q_t = self.noise_schedule.probs_at_t(x, t).log_().clip_(min=-1e6)
            log_p_t = self.noise_schedule.probs_at_t(x_hat, t).log_().clip_(min=-1e6)

            kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(
                -1
            )

            log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
            log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        else:
            logits_nm = self._drop_mask_dim(logits)
            x_hat = logits_nm.softmax(-1).to(dtype)
            x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
            x_nm = self._drop_mask_dim(x)

            alpha_t, beta_pi = self.noise_schedule.get_alpha_betapi(t)
            beta_pi_nm = self._drop_mask_dim(beta_pi)

            probs_q = x_nm.mul(alpha_t.unsqueeze(-1))
            probs_q[..., : beta_pi_nm.shape[-1]].add_(beta_pi_nm.unsqueeze(1))
            probs_p = x_hat.mul(alpha_t.unsqueeze(-1))
            probs_p[..., : beta_pi_nm.shape[-1]].add_(beta_pi_nm.unsqueeze(1))
            log_q_t = probs_q.log_().clip_(min=-1e6)
            log_p_t = probs_p.log_().clip_(min=-1e6)

            kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(
                -1
            )

            z_t_nm = self._remap_no_mask(z_t)
            log_q_zt = log_q_t.gather(-1, z_t_nm.unsqueeze(-1)).squeeze(-1)
            log_p_zt = log_p_t.gather(-1, z_t_nm.unsqueeze(-1)).squeeze(-1)
        log_ratio = log_q_zt - log_p_zt

        is_loss = torch.expm1(log_ratio) - log_ratio
        elbo = elbo_weights * (kl_loss + is_loss)
        loss = ws * (kl_loss + is_loss)

        eps = torch.finfo(loss.dtype).eps
        denom_ws = (ws * attention_mask).sum().clamp_min(eps)
        denom_attn = attention_mask.sum().clamp_min(eps)
        metrics = {
            "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
            "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
            "elbo": (elbo.detach() * attention_mask).sum()
            / denom_attn,
            "loss_weight_mean": (ws.detach() * attention_mask).sum() / denom_attn,
            "loss_weight_min": ws.detach().min(),
            "loss_weight_max": ws.detach().max(),
            "raw_weight_mean": (elbo_weights.detach() * attention_mask).sum()
            / denom_attn,
            "raw_weight_min": elbo_weights.detach().min(),
            "raw_weight_max": elbo_weights.detach().max(),
        }

        if reduction == "tokenmean":
            num_tokens = attention_mask.numel()
            loss = loss.sum() / num_tokens

        return loss, elbo, metrics


class _MaskTokenizerAdapter:
    def __init__(self, base_tokenizer, vocab_size: int, mask_token_id: int):
        self._base = base_tokenizer
        self.mask_token_id = int(mask_token_id)
        self._len = int(vocab_size)

    def __len__(self):
        return self._len


class GIDD(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        self.mask_id, vocab_size = trainer_base.ensure_mask_token(tokenizer)

        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()

        if not getattr(self.config.algo, "time_conditioning", False):
            raise ValueError("GIDD requires algo.time_conditioning=True")

        self._mask_tok = _MaskTokenizerAdapter(
            base_tokenizer=tokenizer, vocab_size=vocab_size, mask_token_id=self.mask_id
        )

        p_uniform = float(getattr(self.config.algo, "p_uniform", None))
        loss_type = str(getattr(self.config.algo, "loss_type", None))
        if loss_type == "gidd_constant_pi":
            self.hybrid_noise = GiddLinearNoise(
                tokenizer=self._mask_tok, p_uniform=p_uniform
            )
            self.loss_fn = GiddLossConstantPi(
                self.config, self._mask_tok, self.hybrid_noise
            )
            self._sample_t_fn = sample_t_constant_pi
        elif loss_type == "gidd":
            gamma = 1.0
            self.hybrid_noise = HybridDiffusion(
                tokenizer=self._mask_tok,
                p_uniform=p_uniform,
                clip_noise=20,
                gamma=gamma,
            )
            self.loss_fn = GiddLoss(self.config, self._mask_tok, self.hybrid_noise)
            self._sample_t_fn = sample_t_hybrid
        else:
            raise ValueError(
                f"Unknown GIDD loss_type={loss_type!r}. Expected one of "
                "{'gidd', 'gidd_constant_pi'}."
            )

        self._loglinear = LogLinear()
        self._val_gen_acc_batches_seen = 0
        self._val_completion_table_rows = []
        self._val_completion_table_columns = None

    def _process_model_input(self, x0, valid_tokens):
        return x0, valid_tokens

    def _process_sigma(self, sigma):
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, batch_size: int):
        eps = float(getattr(self.config.algo, "t_eps", 1e-4))
        return self._sample_t_fn(self.config, batch_size, eps=eps, device=self.device)

    def _sigma_from_alphat(self, alpha_t: torch.Tensor) -> torch.Tensor:
        return -torch.log(alpha_t)

    def _mask_logits_forbidden_classes(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logits[..., self.mask_id] = self.neg_infinity
        return logits

    def _loss(
        self,
        x0,
        valid_tokens,
        current_accumulation_step=None,
        train_mode=False,
        accuracy_tokens=None,
        noise_tokens=None,
    ):
        input_tokens, valid_tokens = self._process_model_input(x0, valid_tokens)
        loss, pred_tokens = self.nll(
            input_tokens,
            current_accumulation_step,
            train_mode,
            valid_tokens=valid_tokens,
            noise_tokens=noise_tokens,
            return_predictions=True,
        )
        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 0] = 0
            valid_tokens[:, 0] = 0
        expected_shape = loss.shape
        if not (
            valid_tokens.shape == expected_shape
            and input_tokens.shape == expected_shape
            and pred_tokens.shape == expected_shape
        ):
            raise ValueError(
                "GIDD shape mismatch: "
                f"loss={loss.shape}, valid={valid_tokens.shape}, "
                f"input={input_tokens.shape}, pred={pred_tokens.shape}"
            )
        if accuracy_tokens is not None and accuracy_tokens.shape != expected_shape:
            raise ValueError(
                "GIDD shape mismatch: "
                f"loss={loss.shape}, accuracy={accuracy_tokens.shape}"
            )
        if noise_tokens is not None and noise_tokens.shape != expected_shape:
            raise ValueError(
                "GIDD shape mismatch: "
                f"loss={loss.shape}, noise={noise_tokens.shape}"
            )

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens

        if accuracy_tokens is None:
            correct_tokens = torch.zeros((), device=loss.device, dtype=loss.dtype)
            num_accuracy_tokens = torch.zeros((), device=loss.device, dtype=loss.dtype)
            correct_samples = torch.zeros((), device=loss.device, dtype=loss.dtype)
            num_accuracy_samples = torch.zeros((), device=loss.device, dtype=loss.dtype)
        else:
            accuracy_mask = accuracy_tokens.to(dtype=torch.bool)
            if self.ignore_bos:
                accuracy_mask = accuracy_mask.clone()
                accuracy_mask[:, 0] = False
            token_correct = pred_tokens == input_tokens
            correct_tokens = (token_correct & accuracy_mask).sum().to(dtype=loss.dtype)
            num_accuracy_tokens = accuracy_mask.sum().to(dtype=loss.dtype)
            has_accuracy = accuracy_mask.any(dim=-1)
            sample_correct = ((~accuracy_mask) | token_correct).all(dim=-1) & has_accuracy
            correct_samples = sample_correct.sum().to(dtype=loss.dtype)
            num_accuracy_samples = has_accuracy.sum().to(dtype=loss.dtype)
        return trainer_base.Loss(
            loss=token_nll,
            nlls=nlls,
            num_tokens=num_tokens,
            correct_tokens=correct_tokens,
            num_accuracy_tokens=num_accuracy_tokens,
            correct_samples=correct_samples,
            num_accuracy_samples=num_accuracy_samples,
        )

    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches
        loss_mask_key = getattr(self.config.algo, "loss_mask_key", "loss_mask")
        accuracy_mask_key = getattr(self.config.algo, "accuracy_mask_key", "accuracy_mask")
        noise_mask_key = getattr(self.config.algo, "noise_mask_key", "noise_mask")

        valid_tokens = batch.get(
            loss_mask_key,
            batch.get("loss_mask", batch["attention_mask"]),
        )
        accuracy_tokens = batch.get(
            accuracy_mask_key,
            batch.get("accuracy_mask", None),
        )
        noise_tokens = None
        if noise_mask_key is not None:
            noise_tokens = batch.get(noise_mask_key, batch.get("noise_mask", None))
        losses = self._loss(
            batch["input_ids"],
            valid_tokens,
            current_accumulation_step,
            train_mode=True,
            accuracy_tokens=accuracy_tokens,
            noise_tokens=noise_tokens,
        )
        self.metrics.update_train(
            losses.nlls,
            losses.num_tokens,
            losses.correct_tokens,
            losses.num_accuracy_tokens,
            losses.correct_samples,
            losses.num_accuracy_samples,
        )
        self.log(
            name="trainer/loss",
            value=losses.loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        return losses.loss

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._val_gen_acc_batches_seen = 0

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        train_acc_token = (
            self.metrics.train_acc_token.compute()
            if getattr(self.metrics.train_acc_token, "weight", 0) > 0
            else torch.tensor(0.0, device=self.device)
        )
        train_acc_sample = (
            self.metrics.train_acc_sample.compute()
            if getattr(self.metrics.train_acc_sample, "weight", 0) > 0
            else torch.tensor(0.0, device=self.device)
        )
        self.log(
            name="train/denoise_acc_token",
            value=train_acc_token,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name="train/denoise_acc_sample",
            value=train_acc_sample,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        loss_mask_key = getattr(self.config.algo, "loss_mask_key", "loss_mask")
        accuracy_mask_key = getattr(self.config.algo, "accuracy_mask_key", "accuracy_mask")
        noise_mask_key = getattr(self.config.algo, "noise_mask_key", "noise_mask")

        valid_tokens = batch.get(
            loss_mask_key,
            batch.get("loss_mask", batch["attention_mask"]),
        )
        accuracy_tokens = batch.get(
            accuracy_mask_key,
            batch.get("accuracy_mask", None),
        )
        noise_tokens = None
        if noise_mask_key is not None:
            noise_tokens = batch.get(noise_mask_key, batch.get("noise_mask", None))
        losses = self._loss(
            batch["input_ids"],
            valid_tokens,
            accuracy_tokens=accuracy_tokens,
            noise_tokens=noise_tokens,
        )
        self.metrics.update_valid(
            losses.nlls,
            losses.num_tokens,
            losses.correct_tokens,
            losses.num_accuracy_tokens,
            losses.correct_samples,
            losses.num_accuracy_samples,
        )
        if self._should_compute_generation_accuracy(accuracy_tokens):
            gen_metrics = self._compute_generation_accuracy(
                batch["input_ids"],
                accuracy_tokens,
                batch.get("attention_mask", None),
            )
            if gen_metrics is not None:
                self.metrics.update_valid_gen(*gen_metrics)
                self._val_gen_acc_batches_seen += 1
        return losses.loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        valid_acc_token = (
            self.metrics.valid_acc_token.compute()
            if getattr(self.metrics.valid_acc_token, "weight", 0) > 0
            else torch.tensor(0.0, device=self.device)
        )
        valid_acc_sample = (
            self.metrics.valid_acc_sample.compute()
            if getattr(self.metrics.valid_acc_sample, "weight", 0) > 0
            else torch.tensor(0.0, device=self.device)
        )
        self.log(
            name="val/denoise_acc_token",
            value=valid_acc_token,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            name="val/denoise_acc_sample",
            value=valid_acc_sample,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if getattr(self.metrics.valid_gen_acc_token, "weight", 0) > 0:
            self.log(
                name="val/gen_acc_token",
                value=self.metrics.valid_gen_acc_token.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if getattr(self.metrics.valid_gen_acc_sample, "weight", 0) > 0:
            self.log(
                name="val/gen_acc_sample",
                value=self.metrics.valid_gen_acc_sample.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def _should_compute_generation_accuracy(self, accuracy_tokens):
        if accuracy_tokens is None:
            return False
        if self.trainer.sanity_checking:
            return False
        if not bool(getattr(self.config.eval, "compute_generation_accuracy", True)):
            return False
        raw_max_batches = getattr(self.config.eval, "generation_accuracy_max_batches", 1)
        max_batches = 1 if raw_max_batches is None else int(raw_max_batches)
        if max_batches <= 0:
            return False
        if self._val_gen_acc_batches_seen >= max_batches:
            return False
        sampler = self._create_sampler()
        return sampler is not None

    @torch.no_grad()
    def _compute_generation_accuracy(self, input_tokens, accuracy_tokens, attention_mask=None):
        method = str(self.config.eval.generation_accuracy_method)
        accuracy_mask = accuracy_tokens.to(device=input_tokens.device, dtype=torch.bool)
        if self.ignore_bos:
            accuracy_mask = accuracy_mask.clone()
            accuracy_mask[:, 0] = False
        has_accuracy = accuracy_mask.any(dim=-1)
        if not has_accuracy.any():
            return None

        num_steps = getattr(self.config.eval, "generation_accuracy_steps", None)
        if num_steps is None:
            num_steps = int(self.config.sampling.steps)
        num_steps = int(num_steps)
        if num_steps <= 0:
            return None

        eps = getattr(self.config.eval, "generation_accuracy_eps", None)
        if eps is None:
            eps = float(getattr(self.config.algo, "t_eps", 1e-5))
        eps = float(eps)

        sampler = self._create_sampler()
        if sampler is None:
            return None
        bsz = input_tokens.shape[0]
        timesteps = torch.linspace(1 - eps, eps, num_steps + 1, device=self.device)
        z_t = self.hybrid_noise.sample_prior((bsz, self.num_tokens)).to(self.device)
        inject_bos = getattr(self.config.sampling, "inject_bos", True)
        if inject_bos:
            z_t[:, 0] = self.tokenizer.bos_token_id
        fixed_mask = ~accuracy_mask
        z_t = torch.where(fixed_mask, input_tokens, z_t)
        ones = torch.ones(bsz, device=self.device)
        for i in range(num_steps):
            t = timesteps[i] * ones
            s = timesteps[i + 1] * ones
            z_t = sampler.compute_posterior(self, z_t, t, s)
            z_t = torch.where(fixed_mask, input_tokens, z_t)
        self._maybe_log_validation_completions(
            input_tokens=input_tokens,
            pred_tokens=z_t,
            accuracy_mask=accuracy_mask,
            has_accuracy=has_accuracy,
            attention_mask=attention_mask,
            method=method,
        )

        if method == "brevo_topo":
            # Topological correctness is sample-level only (answer order is non-unique).
            from ..data.generators.brevo import TopoSortDepthStats

            dataset_cfg = getattr(self.config.data, "dataset_config", None)
            if dataset_cfg is None:
                multi = False
            elif hasattr(dataset_cfg, "get"):
                multi = bool(dataset_cfg.get("multi_token", False))
            else:
                multi = bool(getattr(dataset_cfg, "multi_token", False))
            parser = (
                TopoSortDepthStats.parse_tokens_multi
                if multi
                else TopoSortDepthStats.parse_tokens
            )
            correct_samples = 0
            num_samples = int(has_accuracy.sum().item())
            valid_idx = torch.nonzero(has_accuracy, as_tuple=False).squeeze(-1)
            for idx in valid_idx.tolist():
                pred = z_t[idx].detach().cpu().tolist()
                ok, _, _ = parser(pred)
                if ok:
                    correct_samples += 1
            return (
                None,
                None,
                torch.tensor(float(correct_samples), device=self.device),
                torch.tensor(float(num_samples), device=self.device),
            )

        if method != "exact":
            raise ValueError(
                "Unsupported eval.generation_accuracy_method="
                f"{method!r}. Expected one of: exact, brevo_topo."
            )

        token_correct = (z_t == input_tokens) & accuracy_mask
        sample_correct = ((~accuracy_mask) | (z_t == input_tokens)).all(dim=-1) & has_accuracy

        return (
            token_correct.sum().to(dtype=torch.float32),
            accuracy_mask.sum().to(dtype=torch.float32),
            sample_correct.sum().to(dtype=torch.float32),
            has_accuracy.sum().to(dtype=torch.float32),
        )

    def _split_prefix_completion(self, seq_tokens, completion_mask, attention_mask=None):
        if attention_mask is None:
            valid_mask = torch.ones_like(completion_mask, dtype=torch.bool)
        else:
            valid_mask = attention_mask.to(dtype=torch.bool)

        prefix_mask = (~completion_mask) & valid_mask
        completion_mask = completion_mask & valid_mask

        prefix_ids = seq_tokens[prefix_mask]
        completion_ids = seq_tokens[completion_mask]

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            prefix_ids = prefix_ids[prefix_ids != int(pad_id)]
            completion_ids = completion_ids[completion_ids != int(pad_id)]

        return prefix_ids.tolist(), completion_ids.tolist()

    def _render_token_ids(self, token_ids):
        ids = [int(tok) for tok in token_ids]
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        if callable(convert):
            converted = convert(ids)
            if isinstance(converted, list):
                return " ".join(str(tok) for tok in converted)
        return " ".join(str(tok) for tok in ids)

    def _maybe_log_validation_completions(
        self,
        input_tokens,
        pred_tokens,
        accuracy_mask,
        has_accuracy,
        attention_mask,
        method,
    ):
        if not bool(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions", default=False
            )
        ):
            return
        if self._val_gen_acc_batches_seen != 0:
            return
        if self.trainer is None or self.trainer.sanity_checking:
            return
        if not self.trainer.is_global_zero:
            return
        logger = self.trainer.logger
        if logger is None or not hasattr(logger, "log_table"):
            return

        max_samples = int(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_max_samples", default=8
            )
        )
        if max_samples <= 0:
            return

        parser = None
        if method == "brevo_topo":
            from ..data.generators.brevo import TopoSortDepthStats

            dataset_cfg = getattr(self.config.data, "dataset_config", None)
            if dataset_cfg is None:
                multi = False
            elif hasattr(dataset_cfg, "get"):
                multi = bool(dataset_cfg.get("multi_token", False))
            else:
                multi = bool(getattr(dataset_cfg, "multi_token", False))
            parser = (
                TopoSortDepthStats.parse_tokens_multi
                if multi
                else TopoSortDepthStats.parse_tokens
            )

        valid_idx = torch.nonzero(has_accuracy, as_tuple=False).squeeze(-1).tolist()
        rows = []
        for idx in valid_idx[:max_samples]:
            true_seq = input_tokens[idx].detach().cpu()
            pred_seq = pred_tokens[idx].detach().cpu()
            comp_mask = accuracy_mask[idx].detach().cpu()
            attn = attention_mask[idx].detach().cpu() if attention_mask is not None else None

            prefix_ids, target_completion = self._split_prefix_completion(
                true_seq, comp_mask, attn
            )
            _, predicted_completion = self._split_prefix_completion(pred_seq, comp_mask, attn)

            if parser is not None:
                ok, _, _ = parser(pred_seq.tolist())
                rows.append(
                    [
                        self._render_token_ids(prefix_ids),
                        self._render_token_ids(target_completion),
                        self._render_token_ids(predicted_completion),
                        str(bool(ok)),
                    ]
                )
            else:
                rows.append(
                    [
                        self._render_token_ids(prefix_ids),
                        self._render_token_ids(target_completion),
                        self._render_token_ids(predicted_completion),
                    ]
                )

        if not rows:
            return

        if parser is not None:
            columns = ["prefix", "target_completion", "predicted_completion", "topo_ok"]
        else:
            columns = ["prefix", "target_completion", "predicted_completion"]
        include_step = bool(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_include_step", default=False
            )
        )
        if include_step:
            columns = ["global_step"] + columns
            rows = [[int(self.global_step)] + row for row in rows]

        aggregate = bool(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_aggregate", default=True
            )
        )
        if aggregate:
            if self._val_completion_table_columns != columns:
                self._val_completion_table_columns = columns
                self._val_completion_table_rows = []
            self._val_completion_table_rows.extend(rows)
            max_rows = int(
                omegaconf.OmegaConf.select(
                    self.config, "eval.log_validation_completions_max_rows", default=256
                )
            )
            if max_rows > 0 and len(self._val_completion_table_rows) > max_rows:
                self._val_completion_table_rows = self._val_completion_table_rows[-max_rows:]
            data = self._val_completion_table_rows
        else:
            data = rows

        table_key = str(
            omegaconf.OmegaConf.select(
                self.config, "eval.log_validation_completions_table_key", default="val/completions"
            )
        )
        logger.log_table(
            key=table_key,
            columns=columns,
            data=data,
        )
        self._maybe_update_wandb_summary_table(logger, table_key, columns, data)

    def _maybe_update_wandb_summary_table(self, logger, table_key, columns, data):
        experiment = getattr(logger, "experiment", None)
        if experiment is None:
            return
        summary = getattr(experiment, "summary", None)
        if summary is None:
            return
        try:
            import wandb
        except Exception:
            return

        summary_key = str(
            omegaconf.OmegaConf.select(
                self.config,
                "eval.log_validation_completions_summary_key",
                default=f"{table_key}_latest",
            )
        )
        try:
            summary[summary_key] = wandb.Table(columns=columns, data=data)
        except Exception:
            return

    def nll(
        self,
        input_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
        noise_tokens=None,
        return_predictions=False,
    ):
        t = self._sample_t(input_tokens.shape[0])
        z_t = self.hybrid_noise.sample_zt(input_tokens, t)
        noise_mask_source = noise_tokens if noise_tokens is not None else valid_tokens
        if noise_mask_source is not None:
            noise_mask = noise_mask_source.to(
                device=input_tokens.device, dtype=torch.bool
            )
            z_t = torch.where(noise_mask, z_t, input_tokens)

        alpha_t = self._loglinear.alpha_t(t)
        sigma = self._sigma_from_alphat(alpha_t.unsqueeze(-1))

        sigma = self._process_sigma(sigma)
        logits = self.backbone(z_t, sigma)
        pred_tokens = None
        if return_predictions:
            with torch.no_grad():
                pred_logits = self._mask_logits_forbidden_classes(logits)
                pred_tokens = pred_logits.argmax(dim=-1)

        if valid_tokens is None:
            attention_mask = torch.ones_like(input_tokens, dtype=logits.dtype)
        else:
            attention_mask = valid_tokens.to(device=input_tokens.device, dtype=logits.dtype)

        # Use 'none' reduction to return per-token loss [batch, seq_len]
        # TrainerObjectiveMixin._loss() will then do the tokenmean reduction
        loss, _, metrics = self.loss_fn(
            logits=logits,
            input_ids=input_tokens,
            attention_mask=attention_mask,
            z_t=z_t,
            t=t,
            reduction="none",
        )

        if train_mode and self.training:
            self.log(
                "train/elbo",
                metrics["elbo"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/kl_loss",
                metrics["kl_loss"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/is_loss",
                metrics["is_loss"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/gidd_loss_weight_mean",
                metrics["loss_weight_mean"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/gidd_loss_weight_min",
                metrics["loss_weight_min"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/gidd_loss_weight_max",
                metrics["loss_weight_max"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/gidd_raw_weight_mean",
                metrics["raw_weight_mean"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/gidd_raw_weight_min",
                metrics["raw_weight_min"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                "train/gidd_raw_weight_max",
                metrics["raw_weight_max"].detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

        if return_predictions:
            return loss, pred_tokens
        return loss


__all__ = ["GIDD"]
