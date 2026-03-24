#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}" || exit 1

CONFIG_NAME="${CONFIG_NAME:-none}"
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

hydra_overrides=()

print_usage() {
  cat <<'EOF'
Usage:
  bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh [--config-name NAME|-c NAME] [hydra_override ...]
  bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh [CONFIG_NAME] [hydra_override ...]

Examples:
  bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh loader.num_workers=1
  bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh --config-name gidd_star_graph_d3p3n50_constant_lr_sweep loader.num_workers=1
  bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh --config-name gidd_star_graph_d5p3n50_constant_lr_sweep loader.num_workers=1
  bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh --config-name gidd_star_graph_d2p5n50_constant_lr_sweep loader.num_workers=1
  bash examples/gidd/submit_star_graph_gidd_pooled_slurm.sh --config-name gidd_star_graph_d2p2n50_constant_lr_sweep loader.num_workers=1
EOF
}

config_exists() {
  local name="$1"
  [[ -n "${name}" && -f "${REPO_ROOT}/configs/${name}.yaml" ]]
}

while (($# > 0)); do
  case "$1" in
    --config-name|-c)
      if (($# < 2)) || [[ -z "${2}" ]] || [[ "${2}" == -* ]]; then
        echo "Error: --config-name requires a non-empty value." >&2
        print_usage >&2
        exit 1
      fi
      CONFIG_NAME="$2"
      shift 2
      ;;
    --config-name=*)
      CONFIG_NAME="${1#*=}"
      if [[ -z "${CONFIG_NAME}" ]]; then
        echo "Error: --config-name requires a non-empty value." >&2
        print_usage >&2
        exit 1
      fi
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      hydra_overrides+=("$1")
      shift
      ;;
  esac
done

# Support positional config name when flag is omitted:
#   bash .../submit_star_graph_gidd_pooled_slurm.sh gidd_star_graph_d5p3n50_constant_lr_sweep loader.num_workers=1
if (( ${#hydra_overrides[@]} > 0 )) && [[ "${hydra_overrides[0]}" != *=* ]]; then
  positional_config="${hydra_overrides[0]}"
  if config_exists "${positional_config}"; then
    CONFIG_NAME="${positional_config}"
    hydra_overrides=("${hydra_overrides[@]:1}")
  else
    echo "Error: first positional argument '${positional_config}' is not a valid config name." >&2
    echo "Use --config-name <NAME> or provide an existing config under configs/." >&2
    print_usage >&2
    exit 1
  fi
fi

if ! config_exists "${CONFIG_NAME}"; then
  echo "Error: config '${CONFIG_NAME}' not found at configs/${CONFIG_NAME}.yaml" >&2
  print_usage >&2
  exit 1
fi

# Default temp root for pooled jobs.
# Override with LOCAL_TMPDIR if you need a different path.
LOCAL_TMPDIR="${LOCAL_TMPDIR:-/iopsstor/scratch/cscs/mboronski/.tmp}"
mkdir -p "${LOCAL_TMPDIR}"
TMPDIR="${LOCAL_TMPDIR}"
TMP="${LOCAL_TMPDIR}"
TEMP="${LOCAL_TMPDIR}"
SLURM_TMPDIR="${LOCAL_TMPDIR}"
export TMPDIR TMP TEMP SLURM_TMPDIR

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

wandb_cfg = OmegaConf.to_container(cfg.get("wandb"), resolve=False) or {}
if not isinstance(wandb_cfg, dict):
    wandb_cfg = {}
emit("CFG_WANDB_PROJECT", wandb_cfg.get("project"))
emit("CFG_WANDB_GROUP", wandb_cfg.get("group"))
emit("CFG_WANDB_JOB_TYPE", wandb_cfg.get("job_type"))
emit("CFG_WANDB_NAME", wandb_cfg.get("name"))
emit("CFG_HYDRA_SWEEP_DIR", OmegaConf.select(cfg, "hydra.sweep.dir"))
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
WANDB_PROJECT="${WANDB_PROJECT:-${CFG_WANDB_PROJECT}}"
WANDB_GROUP="${WANDB_GROUP:-${CFG_WANDB_GROUP}}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-${CFG_WANDB_JOB_TYPE}}"
WANDB_NAME="${WANDB_NAME:-${CFG_WANDB_NAME}}"
HYDRA_SWEEP_DIR="${CFG_HYDRA_SWEEP_DIR:-}"

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

for override in "${hydra_overrides[@]}"; do
  cmd+=(--override "${override}")
done

quoted_cmd="$(printf '%q ' "${cmd[@]}")"
tmp_message="Using temp dirs: TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} SLURM_TMPDIR=${SLURM_TMPDIR}"
mem_probe_cmd=$'echo "Memory at start:"; echo "  SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-<unset>} SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-<unset>} SLURM_MEM_PER_GPU=${SLURM_MEM_PER_GPU:-<unset>}"; if command -v free >/dev/null 2>&1; then free -h; fi; awk \'/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}\' /proc/meminfo || true; if [[ -f /sys/fs/cgroup/memory.max ]]; then echo "cgroup: memory.max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true) memory.current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || true)"; elif [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then echo "cgroup: memory.limit_in_bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true) memory.usage_in_bytes=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || true)"; fi'
if [[ -n "${SLURM_ENVIRONMENT}" ]]; then
  wrap_cmd="cd ${REPO_ROOT@Q} && echo ${tmp_message@Q} && bash -lc ${mem_probe_cmd@Q} && TMPDIR=${TMPDIR@Q} TMP=${TMP@Q} TEMP=${TEMP@Q} SLURM_TMPDIR=${SLURM_TMPDIR@Q} srun --environment ${SLURM_ENVIRONMENT@Q} --chdir ${REPO_ROOT@Q} ${quoted_cmd}"
else
  wrap_cmd="cd ${REPO_ROOT@Q} && echo ${tmp_message@Q} && bash -lc ${mem_probe_cmd@Q} && TMPDIR=${TMPDIR@Q} TMP=${TMP@Q} TEMP=${TEMP@Q} SLURM_TMPDIR=${SLURM_TMPDIR@Q} ${quoted_cmd}"
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

sweep_dir_display="${SWEEP_DIR:-${HYDRA_SWEEP_DIR:-<unset>}}"
mem_display="${MEM_GB:-<cluster default>}"
partition_display="${PARTITION:-<unset>}"
account_display="${ACCOUNT:-<unset>}"
qos_display="${QOS:-<unset>}"
constraint_display="${CONSTRAINT:-<unset>}"
slurm_env_display="${SLURM_ENVIRONMENT:-<unset>}"
wandb_project_display="${WANDB_PROJECT:-<unset>}"
wandb_group_display="${WANDB_GROUP:-<unset>}"
wandb_job_type_display="${WANDB_JOB_TYPE:-<unset>}"
wandb_name_display="${WANDB_NAME:-<unset>}"
sweep_limit_display="${SWEEP_LIMIT:-<unset>}"

echo "Submit settings:"
echo "  Config: ${CONFIG_NAME}"
echo "  Account: ${account_display}"
echo "  Partition: ${partition_display}"
echo "  QoS: ${qos_display}"
echo "  Constraint: ${constraint_display}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  Resources: nodes=1 ntasks=1 cpus_per_task=${CPUS_PER_TASK} gpus_per_node=${GPUS_PER_NODE} mem_gb=${mem_display}"
echo "  Sweep: gpus=${SWEEP_GPUS} max_parallel=${SWEEP_MAX_PARALLEL} limit=${sweep_limit_display}"
echo "  Sweep dir: ${sweep_dir_display}"
echo "  W&B project: ${wandb_project_display}"
echo "  W&B group: ${wandb_group_display}"
echo "  W&B job_type: ${wandb_job_type_display}"
echo "  W&B name: ${wandb_name_display}"
echo "  Temp dirs: TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} SLURM_TMPDIR=${SLURM_TMPDIR}"
echo "  Slurm environment: ${slurm_env_display}"
echo "  Logs: ${SBATCH_LOG_DIR}"
if (( ${#hydra_overrides[@]} > 0 )); then
  echo "  Forwarded Hydra overrides: ${hydra_overrides[*]}"
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
echo "Temp dirs: TMPDIR=${TMPDIR} TMP=${TMP} TEMP=${TEMP} SLURM_TMPDIR=${SLURM_TMPDIR}"
if [[ -n "${SLURM_ENVIRONMENT}" ]]; then
  echo "Slurm environment: ${SLURM_ENVIRONMENT}"
fi
if (( ${#hydra_overrides[@]} > 0 )); then
  echo "Forwarded Hydra overrides: ${hydra_overrides[*]}"
fi
