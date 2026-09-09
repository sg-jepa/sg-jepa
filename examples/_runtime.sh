#!/usr/bin/env bash

set -euo pipefail

: "${SGJEPA_ROOT:?SGJEPA_ROOT must point to the sg-jepa checkout}"
SGJEPA_ROOT="$(cd -- "$SGJEPA_ROOT" && pwd -P)"
PYTHON_BIN="${SGJEPA_PYTHON:-python}"
if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
scratch_owner="${USER:-user}"
DATA_ROOT="${SGJEPA_DATA_ROOT:-/scratch/${scratch_owner}/sg-jepa/data}"
OUTPUT_ROOT="${SGJEPA_OUTPUT_ROOT:-${SGJEPA_ROOT}/output}"
MENAGERIE_ROOT="${SGJEPA_MENAGERIE_ROOT:-${SGJEPA_ROOT}/outputs/menagerie_assets}"
CACHE_ROOT="${SGJEPA_CACHE_ROOT:-${SGJEPA_ROOT}/outputs/cache}"
RUN_MODE="${RUN_MODE:-full}"
EXAMPLE_TAG="${EXAMPLE_TAG:-examples-${RUN_MODE}}"

case "$RUN_MODE" in
  full|smoke) ;;
  *) echo "RUN_MODE must be full or smoke, got: $RUN_MODE" >&2; exit 2 ;;
esac

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment does not exist; set SGJEPA_PYTHON=/path/to/python" >&2
  exit 2
fi
if [[ ! -f "$SGJEPA_ROOT/pyproject.toml" ]]; then
  echo "SGJEPA_ROOT is not the public repository: $SGJEPA_ROOT" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SGJEPA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"

cd "$SGJEPA_ROOT"

select_example_task() {
  local task_index="$1"
  case "$task_index" in
    0)
      TASK="square"
      WM_CONFIG="configs/train/square_sg_jepa_gru.yaml"
      FULL_EPISODES=48000
      FULL_TRAIN_EPISODES=8000
      FULL_TEST_EPISODES=40000
      POLICY_CONFIG=""
      ;;
    1)
      TASK="franka_basket"
      WM_CONFIG="configs/train/franka_sg_jepa_gru.yaml"
      FULL_EPISODES=13000
      FULL_TRAIN_EPISODES=8000
      FULL_TEST_EPISODES=5000
      POLICY_CONFIG="examples/configs/franka_policy.yaml"
      ;;
    2)
      TASK="arm_paddle_ball"
      WM_CONFIG="configs/train/paddle_sg_jepa_gru.yaml"
      FULL_EPISODES=9600
      FULL_TRAIN_EPISODES=8000
      FULL_TEST_EPISODES=1600
      POLICY_CONFIG="examples/configs/paddle_policy.yaml"
      ;;
    *) echo "invalid example task index: $task_index" >&2; exit 2 ;;
  esac

  RUN_ROOT="${OUTPUT_ROOT}/${EXAMPLE_TAG}"
  DATA_RUN_ROOT="${DATA_ROOT}/${EXAMPLE_TAG}"
  DATASET_ROOT="${SGJEPA_DATASET_ROOT:-${DATA_RUN_ROOT}/${TASK}}"
  WM_OUTPUT="${SGJEPA_WM_OUTPUT:-${RUN_ROOT}/world_models/${TASK}}"
  DOWNSTREAM_OUTPUT="${SGJEPA_DOWNSTREAM_OUTPUT:-${RUN_ROOT}/downstream/${TASK}}"
  EVALUATION_OUTPUT="${SGJEPA_EVALUATION_OUTPUT:-${RUN_ROOT}/evaluation/${TASK}}"
  export TASK WM_CONFIG POLICY_CONFIG RUN_ROOT DATA_RUN_ROOT DATASET_ROOT
  export WM_OUTPUT DOWNSTREAM_OUTPUT EVALUATION_OUTPUT
}

require_cuda() {
  "$PYTHON_BIN" scripts/verify_examples.py gpu
}

verify_dataset() {
  local episodes="$1"
  local train_episodes="$2"
  local test_episodes="$3"
  "$PYTHON_BIN" scripts/verify_examples.py data \
    --path "$DATASET_ROOT" \
    --task "$TASK" \
    --episodes "$episodes" \
    --train-episodes "$train_episodes" \
    --test-episodes "$test_episodes"
}
