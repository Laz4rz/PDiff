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

export WANDB_PROJECT="${WANDB_PROJECT:-star_graph_gidd_lr_sweep}"
export WANDB_GROUP_PREFIX="${WANDB_GROUP_PREFIX:-star-graph-lr-sweep}"

export MODELS="${MODELS:-tiny small}"
export P_VALUES="${P_VALUES:-0 1}"
export LR_VALUES="${LR_VALUES:-5e-5 1e-4 2e-4}"
# Optional logspace override for LR values:
#   LR_MIN=<float> LR_MAX=<float> LR_LOGSPACE_N=<int>
# If all three are set, LR_VALUES is ignored.
export LR_MIN="${LR_MIN:-}"
export LR_MAX="${LR_MAX:-}"
export LR_LOGSPACE_N="${LR_LOGSPACE_N:-}"

export LR_SCHEDULER="${LR_SCHEDULER:-constant_warmup}"
export MAX_EPOCHS="${MAX_EPOCHS:-50}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
export VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
export PRECISION="${PRECISION:-bf16-mixed}"

STAR_DATA_DIR="${STAR_DATA_DIR:-${REPO_ROOT}/data/star}"
SPEC_FILTER_REGEX="${SPEC_FILTER_REGEX:-}"

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

if [[ -n "${SPEC_FILTER_REGEX}" ]]; then
  if command -v rg >/dev/null 2>&1; then
    mapfile -t SPECS < <(printf "%s\n" "${SPECS[@]}" | rg "${SPEC_FILTER_REGEX}" || true)
  else
    mapfile -t SPECS < <(printf "%s\n" "${SPECS[@]}" | grep -E "${SPEC_FILTER_REGEX}" || true)
  fi
fi

if (( ${#SPECS[@]} == 0 )); then
  echo "No matching train/test pairs found in ${STAR_DATA_DIR}"
  exit 1
fi

IFS=' ' read -r -a MODEL_LIST <<< "${MODELS}"
IFS=' ' read -r -a P_LIST <<< "${P_VALUES}"
LR_SOURCE="explicit"

if [[ -n "${LR_MIN}" || -n "${LR_MAX}" || -n "${LR_LOGSPACE_N}" ]]; then
  if [[ -z "${LR_MIN}" || -z "${LR_MAX}" || -z "${LR_LOGSPACE_N}" ]]; then
    echo "For log-spaced LR sweep set all three: LR_MIN, LR_MAX, LR_LOGSPACE_N."
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to generate log-spaced LR values."
    exit 1
  fi

  mapfile -t LR_LIST < <(
    python3 - <<'PY'
import math
import os
import sys

lr_min = float(os.environ["LR_MIN"])
lr_max = float(os.environ["LR_MAX"])
n = int(os.environ["LR_LOGSPACE_N"])

if lr_min <= 0 or lr_max <= 0:
    print("LR_MIN and LR_MAX must be > 0.", file=sys.stderr)
    sys.exit(1)
if lr_max < lr_min:
    print("LR_MAX must be >= LR_MIN.", file=sys.stderr)
    sys.exit(1)
if n < 1:
    print("LR_LOGSPACE_N must be >= 1.", file=sys.stderr)
    sys.exit(1)

if n == 1:
    values = [lr_min]
else:
    log_min = math.log10(lr_min)
    log_max = math.log10(lr_max)
    step = (log_max - log_min) / (n - 1)
    values = [10 ** (log_min + i * step) for i in range(n)]

for value in values:
    print(f"{value:.12g}")
PY
  )
  LR_SOURCE="logspace(min=${LR_MIN}, max=${LR_MAX}, n=${LR_LOGSPACE_N})"
else
  IFS=' ' read -r -a LR_LIST <<< "${LR_VALUES}"
fi

if (( ${#MODEL_LIST[@]} == 0 )); then
  echo "No models configured. Set MODELS, e.g. 'tiny small'."
  exit 1
fi
if (( ${#P_LIST[@]} == 0 )); then
  echo "No P values configured. Set P_VALUES, e.g. '0 1'."
  exit 1
fi
if (( ${#LR_LIST[@]} == 0 )); then
  echo "No LR values configured. Set LR_VALUES, e.g. '5e-5 1e-4 2e-4'."
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

echo "Starting star-graph GIDD LR sweep with:"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_GROUP_PREFIX=${WANDB_GROUP_PREFIX}"
echo "MODELS=${MODEL_LIST[*]}"
echo "P_VALUES=${P_LIST[*]}"
echo "LR_SOURCE=${LR_SOURCE}"
echo "LR_VALUES=${LR_LIST[*]}"
echo "LR_SCHEDULER=${LR_SCHEDULER}"
if (( ${#SCHEDULER_ARGS[@]} > 0 )); then
  echo "LR_SCHEDULER_OVERRIDES=${SCHEDULER_ARGS[*]}"
fi
echo "MAX_EPOCHS=${MAX_EPOCHS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
echo "VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL}"
echo "PRECISION=${PRECISION}"
echo "STAR_DATA_DIR=${STAR_DATA_DIR}"
if [[ -n "${SPEC_FILTER_REGEX}" ]]; then
  echo "SPEC_FILTER_REGEX=${SPEC_FILTER_REGEX}"
fi
echo "DATASET_SPECS=${SPECS[*]}"

total_runs=$(( ${#SPECS[@]} * ${#MODEL_LIST[@]} * ${#P_LIST[@]} * ${#LR_LIST[@]} ))
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

  for model_name in "${MODEL_LIST[@]}"; do
    for p_uniform in "${P_LIST[@]}"; do
      run_group="${WANDB_GROUP_PREFIX}-${model_name}-d${deg}-n${nodes}-p${path_len}${reverse_tag}-P${p_uniform}"

      for lr_value in "${LR_LIST[@]}"; do
        run_idx=$((run_idx + 1))
        run_name="${run_group}-lr${lr_value}"

        echo
        echo "[${run_idx}/${total_runs}] Starting sweep run:"
        echo "  spec=${spec}"
        echo "  train_file=${train_file}"
        echo "  validation_file=${validation_file}"
        echo "  model=${model_name}"
        echo "  model.length=${model_length} (train_max=${train_max_len}, val_max=${validation_max_len})"
        echo "  p_uniform=${p_uniform}"
        echo "  optim.lr=${lr_value}"
        echo "  wandb.group=${run_group}"
        echo "  wandb.name=${run_name}"

        run_args=(
          data=star_graph
          data.tokenizer_name_or_path=ascii-char
          data.dataset_config.train_file="${train_file}"
          data.dataset_config.validation_file="${validation_file}"
          model="${model_name}"
          algo=gidd
          algo.loss_type=gidd_constant_pi
          algo.p_uniform="${p_uniform}"
          algo.loss_weighting=dynamic
          lr_scheduler="${LR_SCHEDULER}"
          strategy=single-device
          trainer.deterministic=true
          trainer.num_nodes=1
          trainer.devices=1
          trainer.max_epochs="${MAX_EPOCHS}"
          loader.global_batch_size="${GLOBAL_BATCH_SIZE}"
          loader.batch_size="${BATCH_SIZE}"
          loader.eval_batch_size="${EVAL_BATCH_SIZE}"
          trainer.log_every_n_steps=25
          trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
          trainer.limit_train_batches=1.0
          trainer.limit_val_batches=1.0
          callbacks.checkpoint_every_n_steps.save_top_k=0
          callbacks.checkpoint_every_n_steps.save_last=false
          callbacks.checkpoint_monitor.monitor=val/nll
          callbacks.checkpoint_monitor.save_top_k=1
          callbacks.checkpoint_monitor.save_last=true
          trainer.precision="${PRECISION}"
          training.torch_compile=false
          model.length="${model_length}"
          model.max_tokens=-1
          model.dropout=0.0
          optim.lr="${lr_value}"
          optim.weight_decay=0.0
          optim.beta1=0.9
          optim.beta2=0.99
          optim.eps=1e-9
          eval.generate_samples=false
          wandb.save_dir="${WANDB_DIR}"
          wandb.project="${WANDB_PROJECT}"
          wandb.group="${run_group}"
          wandb.job_type=lr-sweep
          wandb.name="${run_name}"
        )
        run_args+=("${SCHEDULER_ARGS[@]}")

        uv run python -u -m discrete_diffusion "${run_args[@]}"
      done
    done
  done
done
