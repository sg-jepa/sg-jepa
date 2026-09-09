#!/usr/bin/env bash
#SBATCH --job-name=sgj-aggregate
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1-2
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
exec "${SGJEPA_ROOT:?}/examples/_run_stage.sh" aggregate "${SLURM_ARRAY_TASK_ID:?}"
