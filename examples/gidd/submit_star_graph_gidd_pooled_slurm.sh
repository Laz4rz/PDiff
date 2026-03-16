#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

CONFIG_NAME="${CONFIG_NAME:-gidd_star_graph_constant_lr_sweep}"

JOB_NAME="${JOB_NAME:-gidd-star-pool}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_GB="${MEM_GB:-64}"
PARTITION="${PARTITION:-normal}"
ACCOUNT="${ACCOUNT:-a137}"
QOS="${QOS:-}"
CONSTRAINT="${CONSTRAINT:-}"
SBATCH_LOG_DIR="${SBATCH_LOG_DIR:-${REPO_ROOT}/outputs/slurm_logs}"
DRY_RUN="${DRY_RUN:-0}"

SWEEP_MAX_PARALLEL="${SWEEP_MAX_PARALLEL:-${GPUS_PER_NODE}}"
SWEEP_GPUS="${SWEEP_GPUS:-}"
if [[ -z "${SWEEP_GPUS}" ]]; then
  last_gpu="$((GPUS_PER_NODE - 1))"
  if (( last_gpu < 0 )); then
    echo "GPUS_PER_NODE must be >= 1"
    exit 1
  fi
  SWEEP_GPUS="$(seq -s, 0 "${last_gpu}")"
fi
SWEEP_LIMIT="${SWEEP_LIMIT:-}"
SWEEP_DIR="${SWEEP_DIR:-}"

mkdir -p "${SBATCH_LOG_DIR}"

cmd=(
  "python"
  scripts/hydra_sweep_gpu_pool.py
  --config-name "${CONFIG_NAME}"
  --gpus "${SWEEP_GPUS}"
  --max-parallel "${SWEEP_MAX_PARALLEL}"
)

if [[ -n "${SWEEP_LIMIT}" ]]; then
  cmd+=(--limit "${SWEEP_LIMIT}")
fi
if [[ -n "${SWEEP_DIR}" ]]; then
  cmd+=(--sweep-dir "${SWEEP_DIR}")
fi

for override in "$@"; do
  cmd+=(--override "${override}")
done

quoted_cmd="$(printf '%q ' "${cmd[@]}")"
wrap_cmd="cd ${REPO_ROOT@Q} && ${quoted_cmd}"

sbatch_args=(
  --parsable
  --export=ALL
  --job-name "${JOB_NAME}"
  --nodes 1
  --ntasks 1
  --cpus-per-task "${CPUS_PER_TASK}"
  --gres "gpu:${GPUS_PER_NODE}"
  --time "${TIME_LIMIT}"
  --output "${SBATCH_LOG_DIR}/%x-%j.out"
  --error "${SBATCH_LOG_DIR}/%x-%j.err"
)

if [[ -n "${MEM_GB}" ]]; then
  sbatch_args+=(--mem "${MEM_GB}G")
fi
if [[ -n "${PARTITION}" ]]; then
  sbatch_args+=(--partition "${PARTITION}")
fi
if [[ -n "${ACCOUNT}" ]]; then
  sbatch_args+=(--account "${ACCOUNT}")
fi
if [[ -n "${QOS}" ]]; then
  sbatch_args+=(--qos "${QOS}")
fi
if [[ -n "${CONSTRAINT}" ]]; then
  sbatch_args+=(--constraint "${CONSTRAINT}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run only. sbatch command:"
  printf 'sbatch '
  printf '%q ' "${sbatch_args[@]}"
  printf -- '--wrap %q\n' "${wrap_cmd}"
  exit 0
fi

job_id="$(sbatch "${sbatch_args[@]}" --wrap "${wrap_cmd}")"

echo "Submitted job ${job_id}"
echo "Config: ${CONFIG_NAME}"
echo "GPUs in allocation: ${GPUS_PER_NODE}"
echo "Sweep GPU pool: ${SWEEP_GPUS}"
echo "Max parallel runs: ${SWEEP_MAX_PARALLEL}"
if (( $# > 0 )); then
  echo "Forwarded Hydra overrides: $*"
fi
