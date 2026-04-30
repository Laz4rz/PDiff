# Generalized Interpolating Discrete Diffusion (GIDD)

Reference: [arXiv:2503.04482](https://arxiv.org/abs/2503.04482)

GIDD generalizes masked diffusion by deriving a new family of interpolating discrete diffusion processes that offer greater flexibility in designing noising processes. By leveraging a novel diffusion ELBO and combining masking with uniform noise, it enables the model to correct its own mistakes and improves sample quality.

## Usage

Train on TinyStories:

```bash
bash examples/gidd/ts_gidd.sh
```

Train on the prefix-completion toy dataset:

```bash
bash examples/gidd/prefix_gidd_constant.sh
```

Run the star-graph constant-weight GIDD LR sweep via Hydra config:

```bash
.venv/bin/python -m discrete_diffusion --config-name gidd_star_graph_constant_lr_sweep
```

Run the same sweep as one Slurm allocation with GPU batching (not one Slurm job per sweep point):

```bash
bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh
```

Preview the exact `sbatch` command without submitting:

```bash
DRY_RUN=1 bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh
```

Optional overrides are forwarded to Hydra:

```bash
bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh seed=1,2,3 lr_log10=range(-5.0,-2.0,1)
```

## Citation

```bibtex
@misc{rutte2025generalized,
      title={Generalized Interpolating Discrete Diffusion}, 
      author={Dimitri von Rütte and Janis Fluri and Yuhui Ding and Antonio Orvieto and Bernhard Schölkopf and Thomas Hofmann},
      year={2025},
      eprint={2503.04482},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
