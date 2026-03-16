#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}" || exit 1

CONFIG_NAME="${CONFIG_NAME:-gidd_star_graph_constant_lr_sweep}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No usable Python interpreter found in PATH (tried ${PYTHON_BIN} and python3)." >&2
    exit 1
  fi
fi

hydra_overrides=("$@")

# Read pooled Slurm defaults from the selected Hydra config.
cfg_exports="$(
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${CONFIG_NAME}" "${hydra_overrides[@]}" <<'PY'
import shlex
import sys
from pathlib import Path

try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing Python dependency for config composition. "
        "Set PYTHON_BIN to an environment with hydra-core and omegaconf installed."
    ) from exc

repo_root = Path(sys.argv[1])
config_name = sys.argv[2]
overrides = sys.argv[3:]
config_dir = repo_root / "configs"

with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(
        config_name=config_name,
        overrides=overrides,
        return_hydra_config=True,
    )

pool_cfg = OmegaConf.to_container(cfg.get("pooled_slurm"), resolve=True) or {}
if not isinstance(pool_cfg, dict):
    raise SystemExit("Expected 'pooled_slurm' to be a mapping in config.")

required = ("job_name", "time_limit", "cpus_per_task", "gpus_per_node", "sbatch_log_dir")
missing = [key for key in required if pool_cfg.get(key) in (None, "")]
if missing:
    raise SystemExit(
        "Missing required pooled_slurm settings in config: " + ", ".join(missing)
    )

def emit(name: str, value) -> None:
    if value is None:
        value = ""
    print(f"{name}={shlex.quote(str(value))}")

emit("CFG_JOB_NAME", pool_cfg.get("job_name"))
emit("CFG_TIME_LIMIT", pool_cfg.get("time_limit"))
emit("CFG_GPUS_PER_NODE", pool_cfg.get("gpus_per_node"))
emit("CFG_CPUS_PER_TASK", pool_cfg.get("cpus_per_task"))
emit("CFG_MEM_GB", pool_cfg.get("mem_gb"))
emit("CFG_PARTITION", pool_cfg.get("partition"))
emit("CFG_ACCOUNT", pool_cfg.get("account"))
emit("CFG_QOS", pool_cfg.get("qos"))
emit("CFG_CONSTRAINT", pool_cfg.get("constraint"))
emit("CFG_SBATCH_LOG_DIR", pool_cfg.get("sbatch_log_dir"))
emit("CFG_SLURM_ENVIRONMENT", pool_cfg.get("slurm_environment"))
emit("CFG_SWEEP_MAX_PARALLEL", pool_cfg.get("sweep_max_parallel"))
emit("CFG_SWEEP_GPUS", pool_cfg.get("sweep_gpus"))
emit("CFG_SWEEP_LIMIT", pool_cfg.get("sweep_limit"))
emit("CFG_SWEEP_DIR", pool_cfg.get("sweep_dir"))
PY
)"
eval "${cfg_exports}"

JOB_NAME="${JOB_NAME:-${CFG_JOB_NAME}}"
TIME_LIMIT="${TIME_LIMIT:-${CFG_TIME_LIMIT}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${CFG_GPUS_PER_NODE}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${CFG_CPUS_PER_TASK}}"
MEM_GB="${MEM_GB:-${CFG_MEM_GB}}"
PARTITION="${PARTITION:-${CFG_PARTITION}}"
ACCOUNT="${ACCOUNT:-${CFG_ACCOUNT}}"
QOS="${QOS:-${CFG_QOS}}"
CONSTRAINT="${CONSTRAINT:-${CFG_CONSTRAINT}}"
SBATCH_LOG_DIR="${SBATCH_LOG_DIR:-${CFG_SBATCH_LOG_DIR}}"
if [[ "${SBATCH_LOG_DIR}" != /* ]]; then
  SBATCH_LOG_DIR="${REPO_ROOT}/${SBATCH_LOG_DIR}"
fi
SLURM_ENVIRONMENT="${SLURM_ENVIRONMENT:-${CFG_SLURM_ENVIRONMENT}}"
SWEEP_MAX_PARALLEL="${SWEEP_MAX_PARALLEL:-${CFG_SWEEP_MAX_PARALLEL}}"
SWEEP_GPUS="${SWEEP_GPUS:-${CFG_SWEEP_GPUS}}"
SWEEP_LIMIT="${SWEEP_LIMIT:-${CFG_SWEEP_LIMIT}}"
SWEEP_DIR="${SWEEP_DIR:-${CFG_SWEEP_DIR}}"

if [[ -z "${SWEEP_GPUS}" ]]; then
  last_gpu="$((GPUS_PER_NODE - 1))"
  if (( last_gpu < 0 )); then
    echo "GPUS_PER_NODE must be >= 1"
    exit 1
  fi
  SWEEP_GPUS="$(seq -s, 0 "${last_gpu}")"
fi
if [[ -z "${SWEEP_MAX_PARALLEL}" ]]; then
  SWEEP_MAX_PARALLEL="${GPUS_PER_NODE}"
fi

mkdir -p "${SBATCH_LOG_DIR}"

POOL_SWEEP_RUNNER="${REPO_ROOT}/scripts/hydra_sweep_gpu_pool.py"
if [[ ! -f "${POOL_SWEEP_RUNNER}" ]]; then
  echo "Expected pooled sweep runner at ${POOL_SWEEP_RUNNER}, but it was not found." >&2
  echo "Set REPO_ROOT to your PDiff checkout before submitting." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  "${POOL_SWEEP_RUNNER}"
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
if [[ -n "${SLURM_ENVIRONMENT}" ]]; then
  wrap_cmd="cd ${REPO_ROOT@Q} && srun --environment ${SLURM_ENVIRONMENT@Q} --chdir ${REPO_ROOT@Q} ${quoted_cmd}"
else
  wrap_cmd="cd ${REPO_ROOT@Q} && ${quoted_cmd}"
fi

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
if [[ -n "${SLURM_ENVIRONMENT}" ]]; then
  echo "Slurm environment: ${SLURM_ENVIRONMENT}"
fi
if (( $# > 0 )); then
  echo "Forwarded Hydra overrides: $*"
fi
