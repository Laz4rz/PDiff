#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export P_UNIFORM="${P_UNIFORM:-1.0}"
export STAR_TRAIN_FILE="${STAR_TRAIN_FILE:-deg_3_path_3_nodes_50_reverse_False_200000_train.txt}"
export STAR_VALIDATION_FILE="${STAR_VALIDATION_FILE:-deg_3_path_3_nodes_50_reverse_False_20000_test.txt}"
# export STAR_TRAIN_FILE="${STAR_TRAIN_FILE:-deg_2_path_2_nodes_50_train_200000.txt}"
# export STAR_VALIDATION_FILE="${STAR_VALIDATION_FILE:-deg_2_path_2_nodes_50_test_20000.txt}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${REPO_ROOT}/wandb_logs"
WANDB_DIR="$(realpath -m "${WANDB_DIR}")"
mkdir -p "${WANDB_DIR}"

export MODEL="${MODEL:-tiny}"

echo "Starting Star-Graph GIDD constant-pi training with:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "P_UNIFORM=${P_UNIFORM}"
echo "STAR_TRAIN_FILE=${STAR_TRAIN_FILE}"
echo "STAR_VALIDATION_FILE=${STAR_VALIDATION_FILE}"
echo "MODEL=${MODEL}"

uv run python -u -m discrete_diffusion \
  data=star_graph \
  data.tokenizer_name_or_path=ascii-char \
  data.dataset_config.train_file="${STAR_TRAIN_FILE}" \
  data.dataset_config.validation_file="${STAR_VALIDATION_FILE}" \
  model="${MODEL}" \
  algo=gidd \
  algo.loss_type=gidd_constant_pi \
  algo.p_uniform="${P_UNIFORM}" \
  algo.loss_weighting=dynamic \
  lr_scheduler=cosine_decay_warmup \
  strategy=single-device \
  trainer.deterministic=true \
  trainer.num_nodes=1 trainer.devices=1 \
  trainer.max_epochs=50 \
  loader.global_batch_size=256 \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  trainer.log_every_n_steps=1 \
  trainer.val_check_interval=500 \
  trainer.limit_train_batches=1.0 \
  trainer.limit_val_batches=1.0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100 \
  trainer.precision=bf16-mixed \
  training.torch_compile=false \
  model.length=32 \
  model.max_tokens=-1 \
  model.dropout=0.0 \
  optim.lr=1e-4 \
  optim.weight_decay=0.0 \
  optim.beta1=0.9 \
  optim.beta2=0.99 \
  optim.eps=1e-9 \
  lr_scheduler.warmup_t=20 \
  eval.generate_samples=false \
  wandb.save_dir="${WANDB_DIR}" \
  wandb.name="${MODEL}-gidd-constant-pi-${P_UNIFORM}" \
  wandb.project=star_graph
