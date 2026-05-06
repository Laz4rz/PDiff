from types import SimpleNamespace

import torch

from discrete_diffusion.sampling.gidd import GIDDAdaptiveSampler


class _LogLinear:
    def alpha_t(self, t):
        return 1.0 - t


class _Tokenizer:
    bos_token_id = None


class _HybridNoise:
    def __init__(self, pi, prior_tokens):
        self.pi = torch.tensor(pi, dtype=torch.float32)
        self.prior_tokens = torch.tensor(prior_tokens, dtype=torch.long)

    def sample_prior(self, shape, *, device=None):
        if device is None:
            device = self.pi.device
        return self.prior_tokens.to(device=device).unsqueeze(0).expand(shape).clone()


class _DummyGIDDModel:
    def __init__(self, hybrid_noise, logits):
        self.device = torch.device("cpu")
        self.num_tokens = 3
        self.vocab_size = 5
        self.mask_id = 4
        self.neg_infinity = -1_000_000.0
        self.tokenizer = _Tokenizer()
        self.hybrid_noise = hybrid_noise
        self._loglinear = _LogLinear()
        self._logits = logits

    def _sigma_from_alphat(self, alpha_t):
        return -torch.log(alpha_t.clamp_min(1e-6))

    def _process_sigma(self, sigma):
        return torch.zeros(sigma.shape[0], device=sigma.device)

    def backbone(self, z_t, sigma, attention_mask=None):
        del sigma, attention_mask
        return self._logits.to(device=z_t.device).expand(z_t.shape[0], -1, -1).clone()


def _config():
    return SimpleNamespace(
        sampling=SimpleNamespace(
            steps=3,
            use_float64=False,
            gidd_adaptive_top_k=1,
            gidd_adaptive_temperature=0.0,
            gidd_adaptive_min_prior_prob=0.0,
        ),
        algo=SimpleNamespace(t_eps=1e-4),
    )


def test_gidd_adaptive_sampler_reduces_to_masked_confidence_unmasking():
    logits = torch.full((1, 3, 5), -8.0)
    logits[:, 0, 1] = 5.0
    logits[:, 1, 2] = 4.0
    logits[:, 2, 3] = 3.0
    model = _DummyGIDDModel(
        hybrid_noise=_HybridNoise(pi=[0.0, 0.0, 0.0, 0.0, 1.0], prior_tokens=[4, 4, 4]),
        logits=logits,
    )
    sampler = GIDDAdaptiveSampler(_config())

    samples = sampler.generate(
        model, num_samples=1, num_steps=2, eps=1e-4, inject_bos=False
    )

    assert samples.tolist() == [[1, 2, 4]]


def test_gidd_adaptive_sampler_no_mask_branch_revises_nonmask_tokens():
    logits = torch.full((1, 3, 5), -8.0)
    logits[:, 0, 0] = 5.0
    logits[:, 1, 3] = 5.0
    logits[:, 2, 2] = 5.0
    model = _DummyGIDDModel(
        hybrid_noise=_HybridNoise(
            pi=[0.25, 0.25, 0.25, 0.25, 0.0], prior_tokens=[0, 1, 2]
        ),
        logits=logits,
    )
    sampler = GIDDAdaptiveSampler(_config())

    samples = sampler.generate(
        model, num_samples=1, num_steps=1, eps=1e-4, inject_bos=False
    )

    assert samples.tolist() == [[0, 3, 2]]
