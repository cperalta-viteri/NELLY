#!/usr/bin/env bash

#SBATCH --job-name=hidra-cv
#SBATCH --array=0-9                  # 10 folds: 0..9
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L40s:1                 # 1 GPU per task (use this on most clusters)
#SBATCH --time=02:00:00              # set when you know; placeholder 8h
#SBATCH --partition=standard         # change to your GPU partition
#SBATCH --mem=128GB                      # or set, e.g. 64G; 0 = all allowed per node on some clusters
#SBATCH --output=output/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=christian.peralta-viteri@uni-wuerzburg.de
#SBATCH --mail-type=ALL


set -euo pipefail

cd ~/drug_repurposing/benchmarking/hidra/
source venv_hidra/bin/activate

mkdir -p logs output

FOLD="$SLURM_ARRAY_TASK_ID"
MODE="${MODE:-train}"

case "$MODE" in
  train)
    python HiDRA_training.py \
      --fold "$FOLD" \
      --type "solid_tumors" \
      --experiment "NBS_cells" \
      -e 20 \
      -o "models"
    ;;
  
  predict)
    python HiDRA_predict.py \
      --fold "$FOLD" \
      --type "solid_tumors" \
      --experiment "NBS_cells" \
      -o "predictions" \
      -m "models_solid_tumors_NBS_cells"
    ;;

esac
