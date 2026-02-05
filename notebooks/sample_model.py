import sys
from pathlib import Path
from typing import Literal

import hydra
import torch
from omegaconf import OmegaConf


def _repo_root() -> Path:
    try:
        here = Path(__file__).resolve()
        return here.parents[1]
    except NameError:
        # Fallback for interactive contexts
        return Path(".").resolve()


ROOT = _repo_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from discrete_diffusion.data import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)


def load_model_from_run(run_dir: Path, ckpt_type: Literal["best", "last"] = "best"):
    ckpt_path = (
        Path(run_dir) / "dummy_checkpoints" / "checkpoints" / f"{ckpt_type}.ckpt"
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt["hyper_parameters"]["config"]
    hydra_cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")

    tokenizer = get_tokenizer(ckpt_cfg)
    algo_target = ckpt_cfg.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)
    model = algo_cls.load_from_checkpoint(
        ckpt_path,
        config=ckpt_cfg,
        tokenizer=tokenizer,
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, ckpt_path, ckpt_cfg, hydra_cfg


NUM_SAMPLES = 8
NUM_STEPS = None  # use model default
RUN_DIR = ROOT / "outputs/roneneldan/TinyStories/2026.02.05/130646"

model, tokenizer, ckpt_path, ckpt_cfg, hydra_cfg = load_model_from_run(RUN_DIR)
print("Checkpoint:", ckpt_path)

samples = model.generate_samples(num_samples=NUM_SAMPLES, num_steps=NUM_STEPS)
texts = tokenizer.batch_decode(samples.detach().cpu(), skip_special_tokens=True)
for i, text in enumerate(texts):
    print(f"--- sample {i} ---")
    print(text)
    print()
