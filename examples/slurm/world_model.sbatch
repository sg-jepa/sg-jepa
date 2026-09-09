#!/usr/bin/env bash
#SBATCH --job-name=sgj-lewm
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=192G
#SBATCH --time=23:50:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
exec "${SGJEPA_ROOT:?}/examples/_run_stage.sh" world-model "${SLURM_ARRAY_TASK_ID:?}"
