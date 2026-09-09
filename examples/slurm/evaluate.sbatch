#!/usr/bin/env bash
#SBATCH --job-name=sgj-eval
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=96G
#SBATCH --time=23:50:00
#SBATCH --array=0-10%8
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
evaluation_index="${SLURM_ARRAY_TASK_ID:?}"

if (( evaluation_index == 0 )); then
  task_index=0
  seed_slot=0
elif (( evaluation_index <= 5 )); then
  task_index=1
  seed_slot="$((evaluation_index - 1))"
elif (( evaluation_index <= 10 )); then
  task_index=2
  seed_slot="$((evaluation_index - 6))"
else
  echo "invalid evaluation array index: $evaluation_index" >&2
  exit 2
fi
exec "${SGJEPA_ROOT:?}/examples/_run_stage.sh" evaluate "$task_index" "$seed_slot"
