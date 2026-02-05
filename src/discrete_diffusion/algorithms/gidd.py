"""GIDD algorithm implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base as trainer_base
from ..noise_schedules import LogLinear
from ..noise_schedules import HybridDiffusion, sample_t as sample_t_hybrid
from ..noise_schedules.gidd_constant_pi import GiddLinearNoise, sample_t as sample_t_constant_pi


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
      self.loss_weighting = getattr(config.algo, 'loss_weighting', 'dynamic')
      self.min_loss_weight = float(getattr(config.algo, 'min_loss_weight', 0.0))
      self.max_loss_weight = float(getattr(config.algo, 'max_loss_weight', 2.0))
    assert self.max_loss_weight > 0, "max_loss_weight must be positive"

    self.mask_id = tokenizer.mask_token_id

  def get_weights(self, t: torch.Tensor, z_t: torch.Tensor, input_ids: torch.Tensor):
    orig_dtype = t.dtype
    t = t.unsqueeze(-1).to(torch.float64)
    t1m = (1 - t)

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

    elbo_weights = is_x * weight_on_x + is_mask * weight_on_m + (1 - is_x - is_mask) * weight_on_u

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

    return (alpha_ratio.to(orig_dtype),
            elbo_weights.to(orig_dtype),
            loss_weights.to(orig_dtype))

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
    is_loss *= 0.0
    elbo = elbo_weights * (kl_loss + is_loss)
    loss = ws * (kl_loss + is_loss)

    eps = torch.finfo(loss.dtype).eps
    denom_ws = (ws * attention_mask).sum().clamp_min(eps)
    metrics = {
      "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
      "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
      "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum().clamp_min(eps),
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
      self.loss_weighting = getattr(config.algo, 'loss_weighting', 'dynamic')
      self.min_loss_weight = float(getattr(config.algo, 'min_loss_weight', 0.0))
      self.max_loss_weight = float(getattr(config.algo, 'max_loss_weight', 2.0))
    assert self.max_loss_weight > 0, "max_loss_weight must be positive"

    self.mask_id = tokenizer.mask_token_id
    self.supports_mask = bool((self.noise_schedule.pi[self.mask_id] > 0).item())

  def _drop_mask_dim(self, x: torch.Tensor) -> torch.Tensor:
    mid = self.mask_id
    return torch.cat([x[..., :mid], x[..., mid + 1:]], dim=-1)

  def _remap_no_mask(self, idx: torch.Tensor) -> torch.Tensor:
    mid = self.mask_id
    return idx - (idx > mid).to(idx.dtype)

  def get_weights(self, t: torch.Tensor, z_t: torch.Tensor, input_ids: torch.Tensor):
    orig_dtype = t.dtype
    t = t.unsqueeze(-1).to(torch.float64)

    alpha_ratio = 1 / (1-t)

    is_x = (z_t == input_ids).to(t.dtype)

    pi_values = self.noise_schedule.gather_pi(input_ids)

    weight_on_not_x = 1 / t
    weight_on_x = pi_values / ( 1 - t + t * pi_values)

    elbo_weights = (is_x * weight_on_x + (1 - is_x) * weight_on_not_x) * alpha_ratio

    loss_weights = elbo_weights.clone()
    if self.loss_weighting == "clip":
      loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
    elif self.loss_weighting == "dynamic":
      log_snr_like = torch.sigmoid(-t).clip(-20, 20)
      x_scale = pi_values * torch.exp(log_snr_like)
      loss_weights = (1 - is_x) * ((1 - is_x) + 2 * is_x) + is_x * x_scale
      loss_weights = loss_weights.clip(self.min_loss_weight, self.max_loss_weight)
    elif self.loss_weighting == "raw":
      loss_weights = elbo_weights

    return (alpha_ratio.to(orig_dtype),
            elbo_weights.to(orig_dtype),
            loss_weights.to(orig_dtype))

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

      kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

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
      probs_q[..., :beta_pi_nm.shape[-1]].add_(beta_pi_nm.unsqueeze(1))
      probs_p = x_hat.mul(alpha_t.unsqueeze(-1))
      probs_p[..., :beta_pi_nm.shape[-1]].add_(beta_pi_nm.unsqueeze(1))
      log_q_t = probs_q.log_().clip_(min=-1e6)
      log_p_t = probs_p.log_().clip_(min=-1e6)

      kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

      z_t_nm = self._remap_no_mask(z_t)
      log_q_zt = log_q_t.gather(-1, z_t_nm.unsqueeze(-1)).squeeze(-1)
      log_p_zt = log_p_t.gather(-1, z_t_nm.unsqueeze(-1)).squeeze(-1)
    log_ratio = log_q_zt - log_p_zt

    is_loss = torch.expm1(log_ratio) - log_ratio
    elbo = elbo_weights * (kl_loss + is_loss)
    loss = ws * (kl_loss + is_loss)

    eps = torch.finfo(loss.dtype).eps
    denom_ws = (ws * attention_mask).sum().clamp_min(eps)
    metrics = {
      "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / denom_ws,
      "is_loss": (ws * is_loss.detach() * attention_mask).sum() / denom_ws,
      "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum().clamp_min(eps),
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

    if not getattr(self.config.algo, 'time_conditioning', False):
      raise ValueError('GIDD requires algo.time_conditioning=True')

    self._mask_tok = _MaskTokenizerAdapter(
      base_tokenizer=tokenizer, vocab_size=vocab_size, mask_token_id=self.mask_id)

    p_uniform = float(getattr(self.config.algo, 'p_uniform', None))
    loss_type = str(getattr(self.config.algo, 'loss_type', None))
    if loss_type == 'gidd_constant_pi':
      self.hybrid_noise = GiddLinearNoise(tokenizer=self._mask_tok, p_uniform=p_uniform)
      self.loss_fn = GiddLossConstantPi(self.config, self._mask_tok, self.hybrid_noise)
      self._sample_t_fn = sample_t_constant_pi
    elif loss_type == 'gidd':
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
        "{'gidd', 'gidd_constant_pi'}.")

    self._loglinear = LogLinear()

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
    eps = float(getattr(self.config.algo, 't_eps', 1e-4))
    return self._sample_t_fn(self.config, batch_size, eps=eps, device=self.device)

  def _sigma_from_alphat(self, alpha_t: torch.Tensor) -> torch.Tensor:
    return -torch.log(alpha_t)

  def _mask_logits_forbidden_classes(self, logits: torch.Tensor) -> torch.Tensor:
    logits = logits.clone()
    logits[..., self.mask_id] = self.neg_infinity
    return logits

  def nll(self, input_tokens,
          current_accumulation_step=None, train_mode=False):
    t = self._sample_t(input_tokens.shape[0])
    z_t = self.hybrid_noise.sample_zt(input_tokens, t)

    alpha_t = self._loglinear.alpha_t(t)
    dalpha_t = self._loglinear.alpha_prime_t(t)
    sigma = self._sigma_from_alphat(alpha_t.unsqueeze(-1))

    sigma = self._process_sigma(sigma)
    logits = self.backbone(z_t, sigma)

    attention_mask = torch.ones_like(input_tokens, dtype=logits.dtype)

    # Use 'none' reduction to return per-token loss [batch, seq_len]
    # TrainerObjectiveMixin._loss() will then do the tokenmean reduction
    loss, elbo, metrics = self.loss_fn(
      logits=logits,
      input_ids=input_tokens,
      attention_mask=attention_mask,
      z_t=z_t,
      t=t,
      reduction='none',
    )

    if train_mode and self.training:
      self.log('train/elbo', metrics['elbo'].item(),
               on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
      self.log('train/kl_loss', metrics['kl_loss'].item(),
               on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
      self.log('train/is_loss', metrics['is_loss'].item(),
               on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

    return loss


__all__ = ['GIDD']
