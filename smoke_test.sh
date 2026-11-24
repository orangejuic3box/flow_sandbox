#!/bin/bash
# Mini ablation: smallest and largest n, one seed, 200 epochs, gifs on.
# Checks that training runs and that the gifs + CSVs get written. Outputs land in the submit dir:
#   mkdir -p $SCRATCH/flow_smoke && cd $SCRATCH/flow_smoke && sbatch ~/flow_toybox/smoke_test.sh
#
#SBATCH --job-name=ablate-smoketest
#SBATCH --account=aip-lazar
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -e

module load StdEnv/2023 python/3.11.5
source ~/activate_venv_ml_dev.sh
export MPLBACKEND=Agg                   # no display on compute nodes

#train a model with 8 measures, check gifs
python -u ~/flow_toybox/run_ablation.py --n 8 --world_seed 0 --epochs 200 --save_gif --out_dir results/smoketest

echo "== gifs (expect 6: epochs 0 40 80 120 160 199)"
ls -l gifs/discrete_flow_matching/measures8/

echo "== csvs"
ls -l results/smoketest/world_seed0/

#use this command to test if the gifs are there
# ls -l $SCRATCH/flow_smoke/gifs/discrete_flow_matching/measures8/
