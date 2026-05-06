from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

import discrete_diffusion.__main__  # noqa: F401
from discrete_diffusion.algorithms.flm import FLM, FMLM, FMLM_TwoModel


class DummyTokenizer:
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    pad_token = "[PAD]"
    mask_token = None
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def __len__(self):
        return 16


def _base_config(algo_name="flm"):
    double_temb = algo_name == "fmlm"
    return OmegaConf.create(
        {
            "neg_infinity_mode": "large-finite",
            "loader": {
                "global_batch_size": 2,
                "batch_size": 2,
                "eval_batch_size": 2,
            },
            "training": {
                "ema": 0,
                "antithetic_sampling": True,
                "sampling_eps": 1e-3,
            },
            "sampling": {
                "predictor": "flow",
                "steps": [2],
                "p_nucleus": 1.0,
                "gamma": 0.0,
            },
            "optim": {"lr": 1e-3},
            "noise": {
                "_target_": "discrete_diffusion.noise_schedules.LogLinear",
                "eps": 0,
            },
            "model": {
                "_target_": "discrete_diffusion.models.dit.DIT",
                "hidden_size": 32,
                "cond_dim": 16,
                "length": 4,
                "n_blocks": 1,
                "n_heads": 4,
                "dropout": 0.0,
                "scale_by_sigma": True,
                "attn_backend": "sdpa",
                "attn_softcap": 50.0,
            },
            "algo": {
                "name": algo_name,
                "backbone": "dit",
                "parameterization": "mean",
                "time_conditioning": True,
                "T": 0,
                "causal_attention": False,
                "ignore_bos": False,
                "t_min": 0.0,
                "t_max": 1.0,
                "double_temb": double_temb,
                "learnable_loss_weighting": False,
                "continuous_inputs": True,
                "diagonal_fraction": 0.5,
                "offdiagonal_sampling": "uniform_diff",
                "set_midpoint": "midpoint",
                "boundary_prob": 32,
                "distillation_method": "PSD",
                "use_mse_loss_psd": False,
                "rescale_offdiag_loss_psd": False,
                "use_ema_for_psd_target": False,
                "initialize_student_from_teacher": False,
                "teacher_path": "",
            },
            "checkpointing": {
                "resume_from_ckpt": False,
                "resume_ckpt_path": None,
            },
        }
    )


def test_flm_configs_compose():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        flm_cfg = hydra.compose(config_name="config", overrides=["algo=flm"])
        fmlm_cfg = hydra.compose(config_name="config", overrides=["algo=fmlm"])
        two_model_cfg = hydra.compose(
            config_name="config", overrides=["algo=fmlm_twomodel"]
        )
        low_cfg = hydra.compose(config_name="config_wikitext2v1_flm_low_compute")
    assert flm_cfg.algo._target_ == "discrete_diffusion.algorithms.flm.FLM"
    assert fmlm_cfg.algo._target_ == "discrete_diffusion.algorithms.flm.FMLM"
    assert (
        two_model_cfg.algo._target_ == "discrete_diffusion.algorithms.flm.FMLM_TwoModel"
    )
    assert low_cfg.data.train == "wikitext2-v1"
    assert low_cfg.data.add_mask_token is False
    assert low_cfg.model._target_ == "discrete_diffusion.models.flm_dit.FLMDIT"


def test_flm_loss_is_finite():
    torch.manual_seed(1)
    model = FLM(_base_config("flm"), DummyTokenizer())
    x = torch.randint(0, len(model.tokenizer), (2, 4))
    valid = torch.ones_like(x)
    loss = model.loss(x, None)
    assert loss.shape == x.shape
    assert torch.isfinite(loss).all()
    aggregate = model._loss(x, valid)
    assert torch.isfinite(aggregate.loss)


@pytest.mark.parametrize("cls,algo_name", [(FLM, "flm"), (FMLM, "fmlm")])
def test_flm_sampling_shapes(cls, algo_name):
    torch.manual_seed(1)
    model = cls(_base_config(algo_name), DummyTokenizer())
    samples = model.generate_samples(num_samples=2, num_steps=[2])
    assert samples.shape == (2, 4)
    assert samples.dtype == torch.long


def test_unsupported_fmlm_variant_has_clear_error():
    cfg = _base_config("fmlm")
    cfg.algo.name = "fmlm_twomodel"
    with pytest.raises(NotImplementedError, match="Only FLM pretraining"):
        FMLM_TwoModel(cfg, DummyTokenizer())
