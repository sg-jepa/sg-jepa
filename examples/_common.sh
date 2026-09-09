#!/usr/bin/env bash

set -euo pipefail

example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "$example_dir/.." && pwd -P)"

run_example() {
  local task="$1"
  shift
  local stage="${1:-}"
  local seed_slot="${2:-0}"
  if [[ -z "$stage" || $# -gt 2 ]]; then
    echo "usage: RUN_MODE=smoke|full bash examples/${task}.sh STAGE [SEED_SLOT]" >&2
    return 2
  fi

  case "$task:$stage" in
    square:data|square:world-model|square:probe|square:evaluate) ;;
    franka_basket:data|franka_basket:world-model|franka_basket:policy) ;;
    franka_basket:evaluate|franka_basket:aggregate) ;;
    arm_paddle_ball:data|arm_paddle_ball:world-model|arm_paddle_ball:policy) ;;
    arm_paddle_ball:evaluate|arm_paddle_ball:aggregate) ;;
    *) echo "stage '$stage' is not valid for $task" >&2; return 2 ;;
  esac
  if [[ "$stage" != "evaluate" && $# -ne 1 ]]; then
    echo "SEED_SLOT is accepted only by the evaluate stage" >&2
    return 2
  fi
  if [[ "$task" == "square" && $# -ne 1 ]]; then
    echo "Square evaluation does not use a seed slot" >&2
    return 2
  fi
  if [[ "$stage" == "aggregate" && "${RUN_MODE:-smoke}" != "full" ]]; then
    echo "aggregate requires RUN_MODE=full and five completed evaluation seed slots" >&2
    return 2
  fi

  export SGJEPA_ROOT="${SGJEPA_ROOT:-$repo_root}"
  if [[ -z "${SGJEPA_PYTHON:-}" ]]; then
    SGJEPA_PYTHON="$(command -v python || true)"
    if [[ -z "$SGJEPA_PYTHON" ]]; then
      echo "python was not found; set SGJEPA_PYTHON=/path/to/python" >&2
      return 2
    fi
    export SGJEPA_PYTHON
  fi
  export RUN_MODE="${RUN_MODE:-smoke}"
  export EXAMPLE_TAG="${EXAMPLE_TAG:-examples-${RUN_MODE}}"
  export SGJEPA_DATA_ROOT="${SGJEPA_DATA_ROOT:-${repo_root}/datasets}"
  export SGJEPA_OUTPUT_ROOT="${SGJEPA_OUTPUT_ROOT:-${repo_root}/output}"
  export SGJEPA_MENAGERIE_ROOT="${SGJEPA_MENAGERIE_ROOT:-${repo_root}/outputs/menagerie_assets}"
  export SGJEPA_CPUS="${SGJEPA_CPUS:-8}"

  exec "$repo_root/examples/_run_stage.sh" "$stage" "$task" "$seed_slot"
}
