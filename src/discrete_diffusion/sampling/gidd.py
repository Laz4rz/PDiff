"""GIDD sampler for hybrid diffusion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical
from .base import Sampler


class GIDDSampler(Sampler):
    """Sampler for GIDD (Generalized Iterative Discrete Diffusion) models."""

    def __init__(self, config, forward_process=None, posterior_sampling=None, min_p=None):
        self.config = config
        if posterior_sampling is None:
            posterior_sampling = getattr(
                config.sampling, "gidd_posterior_sampling", "sample"
            )
        self.posterior_sampling = str(posterior_sampling)
        if self.posterior_sampling not in {"sample", "argmax"}:
            raise ValueError(
                "gidd_posterior_sampling must be one of: sample, argmax. "
                f"Got {self.posterior_sampling!r}."
            )
        self.min_p = None if min_p is None else float(min_p)
        if self.min_p is not None and self.min_p < 0:
            raise ValueError(f"min_p must be non-negative, got {self.min_p}.")

    def _compute_posterior_distribution(self, model, z_t, t, s, attention_mask=None):
        """Compute posterior distribution q(z_s | z_t, x_0) for GIDD.

        Args:
          model: The GIDD model instance.
          z_t: Current noisy samples at time t.
          t: Current timestep.
          s: Next (less noisy) timestep.

        Returns:
          Tuple of posterior probabilities and whether the posterior uses the
          full tokenizer vocabulary including the mask token.
        """
        alpha_t = model._loglinear.alpha_t(t)
        sigma_t = model._sigma_from_alphat(alpha_t.unsqueeze(-1))
        sigma_t = model._process_sigma(sigma_t)
        if attention_mask is None:
            logits = model.backbone(z_t, sigma_t)
        else:
            logits = model.backbone(z_t, sigma_t, attention_mask=attention_mask)

        logits = logits.clone()
        logits[..., model.mask_id] = model.neg_infinity

        pi = getattr(model.hybrid_noise, "pi", None)
        supports_mask = True
        if pi is not None:
            supports_mask = bool((pi[model.mask_id] > 0).item())

        if supports_mask:
            probs = logits.softmax(-1)

            # Get noise schedule values
            q_s = model.hybrid_noise.probs_at_t(probs, s)
            q_t = model.hybrid_noise.probs_at_t(probs, t)
            q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

            alpha_t, beta_pi_t = model.hybrid_noise.get_alpha_betapi(t)
            alpha_s, beta_pi_s = model.hybrid_noise.get_alpha_betapi(s)

            # Compute transition probabilities
            alpha_ts = alpha_t / alpha_s
            beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

            vz_t = F.one_hot(z_t, num_classes=model.vocab_size)
            beta_pi_ts_at_zt = (
                beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
            )
            q_ts = alpha_ts.unsqueeze(1) * vz_t + beta_pi_ts_at_zt
        else:
            mask_id = model.mask_id
            logits_nm = torch.cat(
                [logits[..., :mask_id], logits[..., mask_id + 1 :]], dim=-1
            )
            probs = logits_nm.softmax(-1)

            alpha_t, beta_pi_t = model.hybrid_noise.get_alpha_betapi(t)
            alpha_s, beta_pi_s = model.hybrid_noise.get_alpha_betapi(s)
            beta_pi_t = torch.cat(
                [beta_pi_t[..., :mask_id], beta_pi_t[..., mask_id + 1 :]], dim=-1
            )
            beta_pi_s = torch.cat(
                [beta_pi_s[..., :mask_id], beta_pi_s[..., mask_id + 1 :]], dim=-1
            )

            q_s = probs.mul(alpha_s.unsqueeze(-1))
            q_s[..., : beta_pi_s.shape[-1]].add_(beta_pi_s.unsqueeze(1))
            q_t = probs.mul(alpha_t.unsqueeze(-1))
            q_t[..., : beta_pi_t.shape[-1]].add_(beta_pi_t.unsqueeze(1))

            z_t_nm = z_t - (z_t > mask_id).to(z_t.dtype)
            q_zt = q_t.gather(-1, z_t_nm.unsqueeze(-1))

            # Compute transition probabilities
            alpha_ts = alpha_t / alpha_s
            beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

            vz_t = F.one_hot(z_t_nm, num_classes=model.vocab_size - 1)
            beta_pi_ts_at_zt = (
                beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t_nm.unsqueeze(-1))
            )
            q_ts = alpha_ts.unsqueeze(1) * vz_t + beta_pi_ts_at_zt

        # Compute posterior
        q_st = q_ts * q_s / q_zt

        # Optional: apply minimum probability threshold
        min_p = self.min_p
        if min_p is None:
            min_p = getattr(self.config.sampling, "min_p", 0.0)
        min_p = float(min_p)
        if min_p > 0.0:
            filtered = q_st.masked_fill(q_st < min_p, 0)
            denom = filtered.sum(-1, keepdim=True)
            eps = torch.finfo(q_st.dtype).eps
            q_st = torch.where(denom > eps, filtered / denom.clamp_min(eps), q_st)

        return q_st, supports_mask

    def _sample_from_posterior(self, posterior, supports_mask, model):
        if self.posterior_sampling == "argmax":
            sampled = posterior.argmax(dim=-1)
        else:
            sampled = sample_categorical(posterior)
        if not supports_mask:
            # No-mask branch samples in a compressed vocab with mask removed.
            # Remap sampled ids back to original tokenizer ids.
            sampled = sampled + (sampled >= model.mask_id).to(sampled.dtype)
        return sampled

    def _resamples_nonmask_tokens(self, model):
        pi = getattr(model.hybrid_noise, "pi", None)
        if pi is None:
            return True

        pi = pi.detach()
        nonmask_pi = pi.clone()
        nonmask_pi[model.mask_id] = 0
        return bool((nonmask_pi > 0).any().item())

    def compute_posterior(self, model, z_t, t, s, attention_mask=None):
        """Compute posterior q(z_s | z_t, x_0) for GIDD.

        Args:
          model: The GIDD model instance.
          z_t: Current noisy samples at time t.
          t: Current timestep.
          s: Next (less noisy) timestep.

        Returns:
          Samples from the posterior distribution.
        """
        posterior, supports_mask = self._compute_posterior_distribution(
            model, z_t, t, s, attention_mask=attention_mask
        )
        return self._sample_from_posterior(posterior, supports_mask, model)

    def _compute_update_mask(
        self,
        *,
        step_idx,
        num_steps,
        z_t,
        proposal,
        posterior,
        fixed_mask,
        attention_mask,
        updated_mask,
        mask_id,
        resample_nonmask,
        supports_mask,
    ):
        del (
            step_idx,
            num_steps,
            z_t,
            proposal,
            posterior,
            fixed_mask,
            attention_mask,
            updated_mask,
            mask_id,
            resample_nonmask,
            supports_mask,
        )
        return None

    @torch.no_grad()
    def generate(
        self,
        model,
        *,
        num_samples,
        num_steps,
        eps,
        inject_bos,
        record_steps=False,
        fixed_tokens=None,
        fixed_mask=None,
        attention_mask=None,
    ):
        """Generate samples using GIDD reverse diffusion process.

        Args:
          model: The GIDD model instance.
          num_samples: Number of samples to generate.
          num_steps: Number of denoising steps.
          eps: Minimum timestep value (epsilon).
          inject_bos: Whether to inject BOS token at position 0.
          record_steps: Whether to record intermediate steps.
          fixed_tokens: Optional tokens to keep fixed during generation.
          fixed_mask: Boolean mask selecting positions from fixed_tokens.
          attention_mask: Optional attention mask passed to the backbone.
        Returns:
          Generated token sequences of shape [num_samples, num_tokens], or a
          trajectory of shape [num_steps + 1, num_samples, num_tokens] when
          record_steps=True.
        """
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if eps is None:
            eps = getattr(self.config.algo, "t_eps", 1e-4)

        # Sample from the prior (fully noised)
        z_t = model.hybrid_noise.sample_prior(
            (num_samples, model.num_tokens), device=model.device
        )
        if inject_bos:
            z_t[:, 0] = model.tokenizer.bos_token_id

        if fixed_tokens is not None:
            fixed_tokens = fixed_tokens.to(device=model.device, dtype=z_t.dtype)
            if fixed_tokens.shape != z_t.shape:
                raise ValueError(
                    "fixed_tokens must have shape "
                    f"{tuple(z_t.shape)}, got {tuple(fixed_tokens.shape)}"
                )
            if fixed_mask is None:
                fixed_mask = torch.ones_like(fixed_tokens, dtype=torch.bool)
            else:
                fixed_mask = fixed_mask.to(device=model.device, dtype=torch.bool)
                if fixed_mask.shape != z_t.shape:
                    raise ValueError(
                        "fixed_mask must have shape "
                        f"{tuple(z_t.shape)}, got {tuple(fixed_mask.shape)}"
                    )
            z_t = torch.where(fixed_mask, fixed_tokens, z_t)
        elif fixed_mask is not None:
            raise ValueError("fixed_mask requires fixed_tokens.")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=model.device, dtype=torch.bool)

        # Create timestep schedule from 1-eps to eps
        timesteps = torch.linspace(1 - eps, eps, num_steps + 1, device=model.device)

        if record_steps:
            step_history = [z_t.detach().cpu()]

        updated_mask = torch.zeros_like(z_t, dtype=torch.bool)
        resample_nonmask = self._resamples_nonmask_tokens(model)

        # Iteratively denoise
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(num_samples, device=model.device)
            s = timesteps[i + 1] * torch.ones(num_samples, device=model.device)

            posterior, supports_mask = self._compute_posterior_distribution(
                model, z_t, t, s, attention_mask=attention_mask
            )
            proposal = self._sample_from_posterior(posterior, supports_mask, model)
            update_mask = self._compute_update_mask(
                step_idx=i,
                num_steps=num_steps,
                z_t=z_t,
                proposal=proposal,
                posterior=posterior,
                fixed_mask=fixed_mask,
                attention_mask=attention_mask,
                updated_mask=updated_mask,
                mask_id=model.mask_id,
                resample_nonmask=resample_nonmask,
                supports_mask=supports_mask,
            )
            z_t = proposal if update_mask is None else torch.where(
                update_mask, proposal, z_t
            )
            if update_mask is not None:
                updated_mask |= update_mask

            if fixed_mask is not None:
                z_t = torch.where(fixed_mask, fixed_tokens, z_t)

            if record_steps:
                step_history.append(z_t.detach().cpu())

        if record_steps:
            return torch.stack(step_history, dim=0)

        return z_t


class _GIDDTokenScheduledSampler(GIDDSampler):
    """Base for GIDD samplers that update only selected token positions."""

    order = "left_to_right"

    def __init__(
        self,
        config,
        forward_process=None,
        tokens_per_step=None,
        posterior_sampling=None,
        min_p=None,
    ):
        super().__init__(
            config,
            forward_process=forward_process,
            posterior_sampling=posterior_sampling,
            min_p=min_p,
        )
        if tokens_per_step is None:
            tokens_per_step = getattr(config.sampling, "ordered_tokens_per_step", 1)
        self.tokens_per_step = int(tokens_per_step)
        if self.tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be positive.")

    def _sample_from_posterior(self, posterior, supports_mask, model):
        if supports_mask:
            posterior = posterior.clone()
            posterior[..., model.mask_id] = 0
            eps = torch.finfo(posterior.dtype).eps
            denom = posterior.sum(dim=-1, keepdim=True)

            nonmask = torch.ones(
                posterior.shape[-1], device=posterior.device, dtype=posterior.dtype
            )
            nonmask[model.mask_id] = 0
            uniform = nonmask / nonmask.sum().clamp_min(eps)
            posterior = torch.where(
                denom > eps,
                posterior / denom.clamp_min(eps),
                uniform.view(*((1,) * (posterior.ndim - 1)), -1),
            )
        return super()._sample_from_posterior(posterior, supports_mask, model)

    def _eligible_mask(
        self,
        z_t,
        fixed_mask,
        attention_mask,
        *,
        mask_id,
        resample_nonmask,
    ):
        eligible = torch.ones_like(z_t, dtype=torch.bool)
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "Ordered GIDD samplers require a 2D attention_mask "
                    f"[batch, seq_len], got shape {tuple(attention_mask.shape)}."
                )
            eligible &= attention_mask.to(device=z_t.device, dtype=torch.bool)
        if fixed_mask is not None:
            eligible &= ~fixed_mask.to(device=z_t.device, dtype=torch.bool)
        if not resample_nonmask:
            eligible &= z_t == int(mask_id)
        return eligible

    def _proposed_change_mask(self, proposal, mask_id, resample_nonmask):
        if resample_nonmask:
            return torch.ones_like(proposal, dtype=torch.bool)
        return proposal != int(mask_id)

    def _remaining_eligible_mask(self, eligible, updated_mask):
        if updated_mask is None:
            return eligible

        remaining = eligible & ~updated_mask
        exhausted = eligible.any(dim=-1) & ~remaining.any(dim=-1)
        if bool(exhausted.any()):
            updated_mask[exhausted] &= ~eligible[exhausted]
            remaining = eligible & ~updated_mask
        return remaining

    def _ordered_positions(self, batch_size, seq_len, device, eligible):
        del eligible
        positions = torch.arange(seq_len, device=device)
        if self.order == "right_to_left":
            positions = positions.flip(0)
        elif self.order != "left_to_right":
            raise ValueError(
                f"Unknown token order {self.order!r}. Expected left_to_right or right_to_left."
            )
        return positions.unsqueeze(0).expand(batch_size, -1)

    def _compute_update_mask(
        self,
        *,
        step_idx,
        num_steps,
        z_t,
        proposal,
        posterior,
        fixed_mask,
        attention_mask,
        updated_mask,
        mask_id,
        resample_nonmask,
        supports_mask,
    ):
        del step_idx, num_steps, posterior, supports_mask
        batch_size, seq_len = z_t.shape
        eligible = self._remaining_eligible_mask(
            self._eligible_mask(
                z_t,
                fixed_mask,
                attention_mask,
                mask_id=mask_id,
                resample_nonmask=resample_nonmask,
            ),
            updated_mask,
        )
        eligible &= self._proposed_change_mask(proposal, mask_id, resample_nonmask)
        if not bool(eligible.any()):
            return eligible

        order = self._ordered_positions(
            batch_size=batch_size,
            seq_len=seq_len,
            device=z_t.device,
            eligible=eligible,
        )
        eligible_in_order = eligible.gather(1, order)
        ranks_in_order = eligible_in_order.cumsum(dim=-1)
        selected_in_order = eligible_in_order & (ranks_in_order <= self.tokens_per_step)

        selected = torch.zeros_like(eligible)
        selected.scatter_(1, order, selected_in_order)
        return selected


class GIDDLeftToRightSampler(_GIDDTokenScheduledSampler):
    """GIDD sampler that updates token positions from left to right."""

    order = "left_to_right"


class GIDDRightToLeftSampler(_GIDDTokenScheduledSampler):
    """GIDD sampler that updates token positions from right to left."""

    order = "right_to_left"


class GIDDSingleTokenSampler(_GIDDTokenScheduledSampler):
    """GIDD sampler that greedily updates one eligible token per reverse step."""

    def __init__(self, config, forward_process=None, posterior_sampling=None, min_p=None):
        if posterior_sampling is None:
            posterior_sampling = getattr(
                config.sampling, "gidd_single_token_posterior_sampling", "argmax"
            )
        super().__init__(
            config,
            forward_process=forward_process,
            tokens_per_step=1,
            posterior_sampling=posterior_sampling,
            min_p=min_p,
        )

    def _compute_update_mask(
        self,
        *,
        step_idx,
        num_steps,
        z_t,
        proposal,
        posterior,
        fixed_mask,
        attention_mask,
        updated_mask,
        mask_id,
        resample_nonmask,
        supports_mask,
    ):
        del step_idx, num_steps
        eligible = self._remaining_eligible_mask(
            self._eligible_mask(
                z_t,
                fixed_mask,
                attention_mask,
                mask_id=mask_id,
                resample_nonmask=resample_nonmask,
            ),
            updated_mask,
        )
        eligible &= self._proposed_change_mask(proposal, mask_id, resample_nonmask)
        if not bool(eligible.any()):
            return eligible

        proposal_idx = proposal.unsqueeze(-1)
        if not supports_mask:
            proposal_idx = proposal_idx - (proposal_idx > int(mask_id)).to(
                proposal_idx.dtype
            )
        confidence = posterior.gather(-1, proposal_idx).squeeze(-1)
        confidence = confidence.masked_fill(~eligible, -float("inf"))
        selected_idx = confidence.argmax(dim=-1, keepdim=True)

        selected = torch.zeros_like(eligible)
        selected.scatter_(1, selected_idx, eligible.gather(1, selected_idx))
        return selected


class GIDDAdaptiveSampler(GIDDSampler):
    """Confidence-based adaptive sampler for GIDD models.

    This implements the generalized confidence heuristic

        p_prior(z_t) * (max_x p_theta(x | z_t) - p_theta(z_t | z_t))

    from the Scaling Discrete Diffusion paper. For pure masked GIDD this reduces
    to the standard MDM confidence sampler; for no-mask/uniform priors it can
    revise non-mask tokens.
    """

    def __init__(
        self,
        config,
        forward_process=None,
        top_k=None,
        temperature=None,
        min_prior_prob=None,
    ):
        super().__init__(config, forward_process=forward_process)
        if top_k is None:
            top_k = getattr(config.sampling, "gidd_adaptive_top_k", 1)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("gidd_adaptive_top_k must be positive.")

        if temperature is None:
            temperature = getattr(config.sampling, "gidd_adaptive_temperature", 0.0)
        self.temperature = float(temperature)
        if self.temperature < 0:
            raise ValueError("gidd_adaptive_temperature must be non-negative.")

        if min_prior_prob is None:
            min_prior_prob = getattr(config.sampling, "gidd_adaptive_min_prior_prob", 0.0)
        self.min_prior_prob = float(min_prior_prob)
        if self.min_prior_prob < 0:
            raise ValueError("gidd_adaptive_min_prior_prob must be non-negative.")

    def _supports_mask(self, model):
        pi = getattr(model.hybrid_noise, "pi", None)
        if pi is not None:
            return bool((pi[model.mask_id] > 0).item())

        log_prior = getattr(model.hybrid_noise, "log_prior", None)
        if log_prior is not None:
            return bool((log_prior[model.mask_id].exp() > 0).item())

        return True

    def _drop_mask_dim(self, x, mask_id):
        return torch.cat([x[..., :mask_id], x[..., mask_id + 1 :]], dim=-1)

    def _remap_no_mask(self, x, mask_id):
        return x - (x > int(mask_id)).to(x.dtype)

    def _clean_distribution(self, model, z_t, t, attention_mask=None):
        alpha_t = model._loglinear.alpha_t(t)
        sigma_t = model._sigma_from_alphat(alpha_t.unsqueeze(-1))
        sigma_t = model._process_sigma(sigma_t)

        if attention_mask is None:
            logits = model.backbone(z_t, sigma_t)
        else:
            logits = model.backbone(z_t, sigma_t, attention_mask=attention_mask)

        if getattr(self.config.sampling, "use_float64", False):
            logits = logits.to(torch.float64)
        logits = logits.clone()
        logits[..., model.mask_id] = model.neg_infinity

        supports_mask = self._supports_mask(model)
        if supports_mask:
            clean_logits = logits
        else:
            if bool((z_t == int(model.mask_id)).any()):
                raise ValueError(
                    "No-mask GIDD adaptive sampling received mask tokens in z_t."
                )
            clean_logits = self._drop_mask_dim(logits, int(model.mask_id))

        clean_probs = clean_logits.softmax(dim=-1)
        return clean_logits, clean_probs, supports_mask

    def _prior_prob(self, model, z_t, dtype):
        pi = getattr(model.hybrid_noise, "pi", None)
        if pi is not None:
            prior = pi.to(device=z_t.device, dtype=dtype)
            return prior[z_t]

        log_prior = getattr(model.hybrid_noise, "log_prior", None)
        if log_prior is not None:
            prior = log_prior.to(device=z_t.device, dtype=dtype).exp()
            return prior[z_t]

        return (z_t == int(model.mask_id)).to(dtype=dtype)

    def _predict_tokens(self, clean_logits, clean_probs, supports_mask, mask_id):
        if self.temperature > 0:
            pred = sample_categorical((clean_logits / self.temperature).softmax(dim=-1))
        else:
            pred = clean_probs.argmax(dim=-1)

        if not supports_mask:
            pred = pred + (pred >= int(mask_id)).to(pred.dtype)
        return pred

    def _candidate_mask(self, z_t, fixed_mask, attention_mask, prior_prob):
        eligible = prior_prob > self.min_prior_prob
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "GIDDAdaptiveSampler requires a 2D attention_mask "
                    f"[batch, seq_len], got shape {tuple(attention_mask.shape)}."
                )
            eligible &= attention_mask.to(device=z_t.device, dtype=torch.bool)
        if fixed_mask is not None:
            eligible &= ~fixed_mask.to(device=z_t.device, dtype=torch.bool)
        return eligible

    @torch.no_grad()
    def generate(
        self,
        model,
        *,
        num_samples,
        num_steps,
        eps,
        inject_bos,
        record_steps=False,
        fixed_tokens=None,
        fixed_mask=None,
        attention_mask=None,
    ):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if eps is None:
            eps = getattr(self.config.algo, "t_eps", 1e-4)

        z_t = model.hybrid_noise.sample_prior(
            (num_samples, model.num_tokens), device=model.device
        )
        if inject_bos:
            z_t[:, 0] = model.tokenizer.bos_token_id

        if fixed_tokens is not None:
            fixed_tokens = fixed_tokens.to(device=model.device, dtype=z_t.dtype)
            if fixed_tokens.shape != z_t.shape:
                raise ValueError(
                    "fixed_tokens must have shape "
                    f"{tuple(z_t.shape)}, got {tuple(fixed_tokens.shape)}"
                )
            if fixed_mask is None:
                fixed_mask = torch.ones_like(fixed_tokens, dtype=torch.bool)
            else:
                fixed_mask = fixed_mask.to(device=model.device, dtype=torch.bool)
                if fixed_mask.shape != z_t.shape:
                    raise ValueError(
                        "fixed_mask must have shape "
                        f"{tuple(z_t.shape)}, got {tuple(fixed_mask.shape)}"
                    )
            z_t = torch.where(fixed_mask, fixed_tokens, z_t)
        elif fixed_mask is not None:
            raise ValueError("fixed_mask requires fixed_tokens.")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=model.device, dtype=torch.bool)

        timesteps = torch.linspace(1 - eps, eps, int(num_steps) + 1, device=model.device)

        if record_steps:
            step_history = [z_t.detach().cpu()]

        for i in range(int(num_steps)):
            t = timesteps[i] * torch.ones(num_samples, device=model.device)
            clean_logits, clean_probs, supports_mask = self._clean_distribution(
                model, z_t, t, attention_mask=attention_mask
            )

            if supports_mask:
                curr_idx = z_t
            else:
                curr_idx = self._remap_no_mask(z_t, int(model.mask_id))

            curr_prob = clean_probs.gather(-1, curr_idx.unsqueeze(-1)).squeeze(-1)
            best_prob = clean_probs.max(dim=-1).values
            prior_prob = self._prior_prob(model, z_t, dtype=clean_probs.dtype)
            confidence = prior_prob * (best_prob - curr_prob)

            eligible = self._candidate_mask(
                z_t,
                fixed_mask=fixed_mask,
                attention_mask=attention_mask,
                prior_prob=prior_prob,
            )
            confidence = confidence.masked_fill(~eligible, -float("inf"))
            if not bool(eligible.any()):
                break

            k = min(self.top_k, z_t.shape[1])
            selected_scores, selected_idx = confidence.topk(k=k, dim=-1)
            selected_values = torch.isfinite(selected_scores) & eligible.gather(
                1, selected_idx
            )
            if not bool(selected_values.any()):
                break

            selected = torch.zeros_like(eligible)
            selected.scatter_(1, selected_idx, selected_values)

            pred_tokens = self._predict_tokens(
                clean_logits,
                clean_probs,
                supports_mask=supports_mask,
                mask_id=model.mask_id,
            )
            z_t_next = torch.where(selected, pred_tokens, z_t)

            if fixed_mask is not None:
                z_t_next = torch.where(fixed_mask, fixed_tokens, z_t_next)
            z_t = z_t_next

            if record_steps:
                step_history.append(z_t.detach().cpu())

        if record_steps:
            return torch.stack(step_history, dim=0)

        return z_t
