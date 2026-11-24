#!/bin/bash
# Training-measures ablation on Vulcan: one array task per (world_seed, n) pair.
# Submit from ~/flow_toybox with:   mkdir -p logs && sbatch ablation_job.sh
#
#SBATCH --job-name=ablate-measures
#SBATCH --account=aip-lazar          # your allocation: on vulcan, `ls ~/projects` lists it
#SBATCH --array=0-17                    # must equal len(N_LIST) * len(SEEDS) - 1  (6 * 3 - 1)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00                 # shorten once the smoke test tells you how long n=25 takes
#SBATCH --output=logs/%x_%A_%a.out      # logs/ must exist before you submit, or Slurm drops the log

N_LIST=(1 5 10 15 20 25)
SEEDS=(0 1 2)

i=$SLURM_ARRAY_TASK_ID
N=${N_LIST[$(( i % ${#N_LIST[@]} ))]}
SEED=${SEEDS[$(( i / ${#N_LIST[@]} ))]}

module load python/3.11
source ~/envs/flow/bin/activate
export MPLBACKEND=Agg                   # no display on compute nodes

echo "task $i: n=$N world_seed=$SEED on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# gifs for seed 0 only: gif paths don't include the seed, so tasks with the same n would overwrite each other
GIF_FLAG=""
if [ "$SEED" -eq 0 ]; then GIF_FLAG="--save_gif"; fi

python -u run_ablation.py --n "$N" --world_seed "$SEED" $GIF_FLAG
