#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${REPO_ROOT}/wandb_logs"
WANDB_DIR="$(realpath -m "${WANDB_DIR}")"
mkdir -p "${WANDB_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT:-star_graph_full_runs}"
export MODEL="${MODEL:-tiny}"
export LR_SCHEDULER="${LR_SCHEDULER:-constant_warmup}"
STAR_DATA_DIR="${STAR_DATA_DIR:-${REPO_ROOT}/data/star}"

if [[ ! -d "${STAR_DATA_DIR}" ]]; then
  echo "STAR_DATA_DIR not found: ${STAR_DATA_DIR}"
  exit 1
fi

spec_from_filename() {
  local filename="$1"
  local spec=""
  if [[ "${filename}" =~ ^(deg_[^_]+_path_[^_]+_nodes_[^_]+(_reverse_[^_]+)?)_ ]]; then
    spec="${BASH_REMATCH[1]}"
  fi
  echo "${spec}"
}

longest_entry_len() {
  local file_path="$1"
  awk 'length > max {max = length} END {print max + 0}' "${file_path}"
}

declare -A TRAIN_FILE_BY_SPEC
declare -A TEST_FILE_BY_SPEC

mapfile -t STAR_FILES < <(find "${STAR_DATA_DIR}" -maxdepth 1 -type f -name "*.txt" -printf "%f\n" | sort)

if (( ${#STAR_FILES[@]} == 0 )); then
  echo "No star files found in ${STAR_DATA_DIR}"
  exit 1
fi

for filename in "${STAR_FILES[@]}"; do
  spec="$(spec_from_filename "${filename}")"
  if [[ -z "${spec}" ]]; then
    continue
  fi

  if [[ "${filename}" == *"_train"* ]]; then
    TRAIN_FILE_BY_SPEC["${spec}"]="${filename}"
  fi

  if [[ "${filename}" == *"_test"* ]]; then
    TEST_FILE_BY_SPEC["${spec}"]="${filename}"
  fi
done

mapfile -t SPECS < <(
  for spec in "${!TRAIN_FILE_BY_SPEC[@]}"; do
    if [[ -n "${TEST_FILE_BY_SPEC[${spec}]+x}" ]]; then
      echo "${spec}"
    fi
  done | sort
)

if (( ${#SPECS[@]} == 0 )); then
  echo "No matching train/test pairs found in ${STAR_DATA_DIR}"
  exit 1
fi

IFS=' ' read -r -a P_VALUES <<< "${P_VALUES:-0 1}"
if (( ${#P_VALUES[@]} == 0 )); then
  echo "No P values configured. Set P_VALUES, e.g. '0 1'."
  exit 1
fi

SCHEDULER_ARGS=()
case "${LR_SCHEDULER}" in
  cosine_decay_warmup)
    LR_WARMUP_T="${LR_WARMUP_T:-20}"
    SCHEDULER_ARGS+=(lr_scheduler.warmup_t="${LR_WARMUP_T}")
    ;;
  constant_warmup)
    if [[ -n "${LR_NUM_WARMUP_STEPS:-}" ]]; then
      SCHEDULER_ARGS+=(lr_scheduler.num_warmup_steps="${LR_NUM_WARMUP_STEPS}")
    fi
    ;;
  step_scheduler)
    if [[ -n "${LR_STEP_WARMUP_STEPS:-}" ]]; then
      SCHEDULER_ARGS+=(lr_scheduler.lr_lambda.warmup_steps="${LR_STEP_WARMUP_STEPS}")
    fi
    if [[ -n "${LR_STEP_N_HALVE_STEPS:-}" ]]; then
      SCHEDULER_ARGS+=(lr_scheduler.lr_lambda.n_halve_steps="${LR_STEP_N_HALVE_STEPS}")
    fi
    ;;
  constant)
    ;;
  *)
    echo "Unknown LR_SCHEDULER='${LR_SCHEDULER}'."
    echo "Supported: constant, constant_warmup, cosine_decay_warmup, step_scheduler"
    exit 1
    ;;
esac

echo "Starting Star-Graph full GIDD constant-pi runs with:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "MODEL=${MODEL}"
echo "LR_SCHEDULER=${LR_SCHEDULER}"
echo "CHECKPOINT_MODE=best(val/nll)+last"
if (( ${#SCHEDULER_ARGS[@]} > 0 )); then
  echo "LR_SCHEDULER_OVERRIDES=${SCHEDULER_ARGS[*]}"
fi
echo "STAR_DATA_DIR=${STAR_DATA_DIR}"
echo "P_VALUES=${P_VALUES[*]}"
echo "DATASET_SPECS=${SPECS[*]}"

total_runs=$(( ${#SPECS[@]} * ${#P_VALUES[@]} ))
run_idx=0

for spec in "${SPECS[@]}"; do
  train_file="${TRAIN_FILE_BY_SPEC[${spec}]}"
  validation_file="${TEST_FILE_BY_SPEC[${spec}]}"
  train_path="${STAR_DATA_DIR}/${train_file}"
  validation_path="${STAR_DATA_DIR}/${validation_file}"

  deg="unknown"
  path_len="unknown"
  nodes="unknown"
  reverse_tag=""
  if [[ "${spec}" =~ deg_([0-9]+)_path_([0-9]+)_nodes_([0-9]+) ]]; then
    deg="${BASH_REMATCH[1]}"
    path_len="${BASH_REMATCH[2]}"
    nodes="${BASH_REMATCH[3]}"
  fi
  if [[ "${spec}" =~ _reverse_([^_]+) ]]; then
    reverse_tag="-reverse-${BASH_REMATCH[1]}"
  fi

  train_max_len="$(longest_entry_len "${train_path}")"
  validation_max_len="$(longest_entry_len "${validation_path}")"
  if (( train_max_len >= validation_max_len )); then
    model_length="${train_max_len}"
  else
    model_length="${validation_max_len}"
  fi
  if (( model_length < 1 )); then
    model_length=1
  fi

  for p_uniform in "${P_VALUES[@]}"; do
    run_idx=$((run_idx + 1))
    run_name="${MODEL}-cgiddraw-d${deg}-n${nodes}-p${path_len}-P${p_uniform}"

    echo
    echo "[${run_idx}/${total_runs}] Starting run:"
    echo "  spec=${spec}"
    echo "  train_file=${train_file}"
    echo "  validation_file=${validation_file}"
    echo "  model.length=${model_length} (train_max=${train_max_len}, val_max=${validation_max_len})"
    echo "  p_uniform=${p_uniform}"
    echo "  wandb.name=${run_name}"

    run_args=(
      data=star_graph
      data.tokenizer_name_or_path=ascii-char
      data.dataset_config.train_file="${train_file}"
      data.dataset_config.validation_file="${validation_file}"
      model="${MODEL}"
      algo=gidd
      algo.loss_type=gidd_constant_pi
      algo.p_uniform="${p_uniform}"
      algo.loss_weighting=dynamic
      lr_scheduler="${LR_SCHEDULER}"
      strategy=single-device
      trainer.deterministic=true
      trainer.num_nodes=1
      trainer.devices=1
      trainer.max_epochs=50
      loader.global_batch_size=256
      loader.batch_size=256
      loader.eval_batch_size=256
      trainer.log_every_n_steps=1
      trainer.val_check_interval=500
      trainer.limit_train_batches=1.0
      trainer.limit_val_batches=1.0
      callbacks.checkpoint_every_n_steps.save_top_k=0
      callbacks.checkpoint_every_n_steps.save_last=false
      callbacks.checkpoint_monitor.monitor=val/nll
      callbacks.checkpoint_monitor.save_top_k=1
      callbacks.checkpoint_monitor.save_last=true
      trainer.precision=bf16-mixed
      training.torch_compile=false
      model.length="${model_length}"
      model.max_tokens=-1
      model.dropout=0.0
      optim.lr=1e-4
      optim.weight_decay=0.0
      optim.beta1=0.9
      optim.beta2=0.99
      optim.eps=1e-9
      eval.generate_samples=false
      wandb.save_dir="${WANDB_DIR}"
      wandb.project="${WANDB_PROJECT}"
      wandb.group="star-graph-full-runs"
      wandb.name="${run_name}"
    )
    run_args+=("${SCHEDULER_ARGS[@]}")

    uv run python -u -m discrete_diffusion "${run_args[@]}"
  done
done
