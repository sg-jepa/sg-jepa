#!/usr/bin/env bash

set -euo pipefail

run_mode="${1:-}"
through_stage="${2:-world-model}"
case "$run_mode" in
  smoke|full) ;;
  *) echo "usage: bash examples/submit_slurm.sh smoke|full [world-model|all]" >&2; exit 2 ;;
esac
case "$through_stage" in
  world-model|all) ;;
  *) echo "stage must be world-model or all" >&2; exit 2 ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
if [[ "$run_mode" == "smoke" ]]; then
  default_tag="smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  data_partition=rtx-devel
  compute_partition=b200-devel
  data_time=04:00:00
  compute_time=04:00:00
else
  default_tag=examples-full
  data_partition=rtx-batch
  compute_partition=b200-batch
  data_time=23:50:00
  compute_time=23:50:00
fi
example_tag="${EXAMPLE_TAG:-$default_tag}"
scratch_owner="${USER:-user}"
log_root="${SGJEPA_LOG_ROOT:-/scratch/${scratch_owner}/sg-jepa/logs}"
mkdir -p "$log_root" "${SGJEPA_DATA_ROOT:-/scratch/${scratch_owner}/sg-jepa/data}"

job_exports="ALL,SGJEPA_ROOT=${repo_root},RUN_MODE=${run_mode},EXAMPLE_TAG=${example_tag}"
if [[ -n "${SGJEPA_PYTHON:-}" ]]; then
  job_exports="${job_exports},SGJEPA_PYTHON=${SGJEPA_PYTHON}"
fi
if [[ -n "${SMOKE_EPISODES:-}" ]]; then
  job_exports="${job_exports},SMOKE_EPISODES=${SMOKE_EPISODES}"
fi

data_receipt="$(sbatch --parsable \
  --partition="$data_partition" \
  --time="$data_time" \
  --output="${log_root}/%x-%A_%a.out" \
  --chdir="$repo_root" \
  --export="$job_exports" \
  "$repo_root/examples/slurm/data.sbatch")"
data_job="${data_receipt%%;*}"

world_receipt="$(sbatch --parsable \
  --partition="$compute_partition" \
  --time="$compute_time" \
  --output="${log_root}/%x-%A_%a.out" \
  --chdir="$repo_root" \
  --dependency="aftercorr:${data_job}" \
  --export="$job_exports" \
  "$repo_root/examples/slurm/world_model.sbatch")"
world_job="${world_receipt%%;*}"

echo "EXAMPLE_TAG=$example_tag"
echo "DATA_JOB_ID=$data_job"
echo "WORLD_MODEL_JOB_ID=$world_job"

if [[ "$through_stage" == "world-model" ]]; then
  exit 0
fi

downstream_receipt="$(sbatch --parsable \
  --partition="$compute_partition" \
  --time="$compute_time" \
  --output="${log_root}/%x-%A_%a.out" \
  --chdir="$repo_root" \
  --dependency="aftercorr:${world_job}" \
  --export="$job_exports" \
  "$repo_root/examples/slurm/downstream.sbatch")"
downstream_job="${downstream_receipt%%;*}"
echo "DOWNSTREAM_JOB_ID=$downstream_job"

if [[ "$run_mode" == "smoke" ]]; then
  evaluation_array="0,1,6%3"
else
  evaluation_array="0-10%8"
fi
evaluation_receipt="$(sbatch --parsable \
  --partition="$compute_partition" \
  --time="$compute_time" \
  --array="$evaluation_array" \
  --output="${log_root}/%x-%A_%a.out" \
  --chdir="$repo_root" \
  --dependency="afterok:${downstream_job}" \
  --export="$job_exports" \
  "$repo_root/examples/slurm/evaluate.sbatch")"
evaluation_job="${evaluation_receipt%%;*}"
echo "EVALUATION_JOB_ID=$evaluation_job"

if [[ "$run_mode" == "full" ]]; then
  aggregate_receipt="$(sbatch --parsable \
    --partition="$compute_partition" \
    --time=00:30:00 \
    --output="${log_root}/%x-%A_%a.out" \
    --chdir="$repo_root" \
    --dependency="afterok:${evaluation_job}" \
    --export="$job_exports" \
    "$repo_root/examples/slurm/aggregate.sbatch")"
  echo "AGGREGATE_JOB_ID=${aggregate_receipt%%;*}"
fi
