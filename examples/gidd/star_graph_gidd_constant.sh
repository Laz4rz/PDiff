#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export P_UNIFORM=0.0

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${REPO_ROOT}/wandb_logs"
WANDB_DIR="$(realpath -m "${WANDB_DIR}")"
mkdir -p "${WANDB_DIR}"

echo "Starting Star-Graph GIDD constant-pi training with:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "P_UNIFORM=${P_UNIFORM}"

uv run python -u -m discrete_diffusion \
  data=star_graph \
  model=tiny \
  algo=gidd \
  algo.loss_type=gidd_constant_pi \
  algo.p_uniform="${P_UNIFORM}" \
  algo.loss_weighting=dynamic \
  lr_scheduler=cosine_decay_warmup \
  strategy=ddp \
  trainer.deterministic=false \
  trainer.num_nodes=1 trainer.devices=4 \
  trainer.max_epochs=1 \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=4 \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval=1000 \
  trainer.limit_train_batches=0.05 \
  trainer.limit_val_batches=0.05 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
  trainer.precision=bf16-mixed \
  training.torch_compile=false \
  model.length=256 \
  model.max_tokens=256 \
  model.dropout=0.0 \
  data.dataset_config.train_dataset_size=50000 \
  data.dataset_config.validation_dataset_size=5000 \
  data.dataset_config.context_window=100 \
  data.dataset_config.nvertices=20 \
  data.dataset_config.ndistractors=5 \
  data.dataset_config.max_attention=7 \
  data.dataset_config.instruction='Given a directed graph where each node has exactly one outgoing edge, represented by a list of {node, target} pairs, infer the one-step function and answer the query.' \
  data.dataset_config.train_seed=42 \
  data.dataset_config.validation_seed=41 \
  optim.lr=5e-4 \
  optim.weight_decay=0.02 \
  optim.beta1=0.9 \
  optim.beta2=0.99 \
  optim.eps=1e-9 \
  lr_scheduler.warmup_t=100 \
  wandb.save_dir="${WANDB_DIR}" \
  wandb.name="tiny-gidd-constant-pi-${P_UNIFORM}" \
  wandb.project=star-graph-gidd
