"""FLM-specific DiT backbone.

This module keeps source-compatible FLM/FMLM behavior out of the generic PDiff
DiT implementation. The public module/parameter structure mirrors the regular
DiT closely enough for checkpoint and parity checks, while the FLM-only details
live here: raw continuous input embedding, double time embeddings, learnable
loss weighting, and source-style attention softcapping.
"""

from __future__ import annotations

import math

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:  # flash-attn is optional.
    import flash_attn
except (ImportError, RuntimeError):
    flash_attn = None  # type: ignore

from .common import (
    DDiTBlockCausal,
    DDiTFinalLayer,
    LayerNorm,
    Rotary,
    TimestepEmbedder,
    apply_rotary_pos_emb_torchscript,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
    sdpa_attention_masked,
)


class FLMEmbeddingLayer(nn.Module):
    """Embedding layer for FLM continuous simplex/noise states."""

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            x.float(),
            self.embedding.float(),
        ).to(x.dtype)


class FLMLearnableLossWeighting(nn.Module):
    def __init__(self, cond_dim: int, is_flow: bool = True, hidden_dim: int = 128):
        super().__init__()
        self.s_embed = TimestepEmbedder(cond_dim)
        self.t_embed = None if is_flow else TimestepEmbedder(cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.s_embed(s)
        if t is not None and self.t_embed is not None:
            emb = emb + self.t_embed(t)
        return self.mlp(emb).squeeze(-1)


class FLMDDiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        adaLN,
        cond_dim=None,
        mlp_ratio=4,
        dropout=0.1,
        attn_softcap: float = 50.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN
        self.attn_softcap = attn_softcap

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def _softcapped_attention(self, qkv):
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        head_dim = q.shape[-1]
        attn_weights = torch.einsum("bhid,bhjd->bhij", q / math.sqrt(head_dim), k)
        if self.attn_softcap > 0:
            attn_weights = self.attn_softcap * torch.tanh(
                attn_weights / self.attn_softcap
            )
        attn_probs = torch.softmax(attn_weights, dim=-1)
        return torch.einsum("bhij,bhjd->bhid", attn_probs, v).transpose(1, 2)

    def _apply_attention(self, qkv, rotary_cos_sin, attn_mask, use_jvp_attn=False):
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb_torchscript(
            qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
        )
        if attn_mask is not None:
            q, k, v = [x.squeeze(2) for x in qkv.chunk(3, dim=2)]
            return sdpa_attention_masked(q, k, v, attn_mask, causal=False)

        if not use_jvp_attn and flash_attn is not None and qkv.is_cuda:
            try:
                return flash_attn.flash_attn_qkvpacked_func(
                    qkv,
                    0.0,
                    causal=False,
                    softcap=float(self.attn_softcap),
                ).flatten(-2)
            except (TypeError, RuntimeError):
                pass
        x = self._softcapped_attention(qkv)
        return rearrange(x, "b s h d -> b s (h d)")

    def forward(self, x, rotary_cos_sin, c=None, attn_mask=None, use_jvp_attn=False):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            )
            x = modulate_fused(x, shift_msa, scale_msa)

        qkv = rearrange(
            self.attn_qkv(x),
            "b s (three h d) -> b s three h d",
            three=3,
            h=self.n_heads,
        )
        x = self._apply_attention(qkv, rotary_cos_sin, attn_mask, use_jvp_attn)

        if self.adaLN:
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, gate_msa, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
                None,
                gate_mlp,
                x,
                self.dropout,
            )
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout
            )
        return x


class FLMDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """Source-compatible DiT backbone for FLM/FMLM algorithms."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.use_time_conditioning = bool(
            getattr(config.model, "use_time_conditioning", True)
        )
        self.vocab_embed = FLMEmbeddingLayer(dim, vocab_size)
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
            self.sigma_map_prime = (
                TimestepEmbedder(cond_dim)
                if bool(getattr(config.algo, "double_temb", False))
                else None
            )
        self.rotary_emb = Rotary(dim // config.model.n_heads)

        if bool(getattr(config.algo, "learnable_loss_weighting", False)):
            self.learnable_loss_weighting = FLMLearnableLossWeighting(cond_dim=cond_dim)

        attn_softcap = float(getattr(config.model, "attn_softcap", 50.0))
        blocks = []
        for _ in range(config.model.n_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim, n_heads=config.model.n_heads, dropout=config.model.dropout
                )
            else:
                block = FLMDDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout,
                    attn_softcap=attn_softcap,
                )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN,
        )
        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(
        self, x, sigma, sigma_prime=None, attention_mask=None, use_jvp_attn=False
    ):
        x = self.vocab_embed(x)
        if self.causal:
            t_cond = None
        else:
            if not self.use_time_conditioning:
                sigma = torch.zeros_like(sigma)
            t_emb = self.sigma_map(sigma)
            if sigma_prime is not None:
                if self.sigma_map_prime is not None:
                    t_emb = t_emb + self.sigma_map_prime(sigma_prime)
                else:
                    t_emb = t_emb + self.sigma_map(sigma_prime)
            t_cond = F.silu(t_emb)

        rotary_cos_sin = self.rotary_emb(x)
        attn_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            attn_mask = attention_mask[:, None, :].expand(-1, x.shape[1], -1)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(
                    x,
                    rotary_cos_sin,
                    c=t_cond,
                    attn_mask=attn_mask,
                    use_jvp_attn=use_jvp_attn,
                )
            x = self.output_layer(x, c=t_cond)
        return x


__all__ = [
    "FLMDIT",
    "FLMDDiTBlock",
    "FLMEmbeddingLayer",
    "FLMLearnableLossWeighting",
]
