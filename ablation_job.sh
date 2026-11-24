#!/bin/bash
# Training-measures ablation on Vulcan: one array task per (world_seed, n) pair.
# logs/, results/ and gifs/ are written in the directory you submit from, so submit from $SCRATCH:
#   mkdir -p $SCRATCH/flow_toybox/logs && cd $SCRATCH/flow_toybox && sbatch ~/flow_toybox/ablation_job.sh
#
# Run smoke_test.sh first to check training and gif saving on a short run.
#
#SBATCH --job-name=ablate-measures
#SBATCH --array=0-17                # must equal len(N_LIST) * len(SEEDS) - 1  (6 * 3 - 1)

#SBATCH --account=aip-lazar
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

#SBATCH --output=logs/%x_%A_%a.out      # logs/ must exist before you submit, or Slurm drops the log

N_LIST=(1 5 10 15 20 25)
SEEDS=(0 1 2)

EPOCHS=${EPOCHS:-20000}

i=$SLURM_ARRAY_TASK_ID
N=${N_LIST[$(( i % ${#N_LIST[@]} ))]}
SEED=${SEEDS[$(( i / ${#N_LIST[@]} ))]}

module load StdEnv/2023 python/3.11.5
source ~/activate_venv_ml_dev.sh

export MPLBACKEND=Agg                   # no display on compute nodes

echo "task $i: n=$N world_seed=$SEED epochs=$EPOCHS on $(hostname), writing to $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# gifs for seed 0 only: gif paths don't include the seed, so tasks with the same n would overwrite each other
GIF_FLAG=""
if [ "$SEED" -eq 0 ]; then GIF_FLAG="--save_gif"; fi

python -u ~/flow_toybox/run_ablation.py --n "$N" --world_seed "$SEED" --epochs "$EPOCHS" $GIF_FLAG
