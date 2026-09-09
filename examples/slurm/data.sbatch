#!/usr/bin/env bash
#SBATCH --job-name=sgj-data
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=96G
#SBATCH --time=23:50:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
exec "${SGJEPA_ROOT:?}/examples/_run_stage.sh" data "${SLURM_ARRAY_TASK_ID:?}"
