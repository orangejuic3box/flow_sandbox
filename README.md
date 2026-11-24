## 4Mags
Always run code in venv:
```bash
source ~/activate_venv_ml_dev.sh   # module purge; python/3.10 + cuda/12.6; activate ~/ml-env
```

To submit a job, run on vulcan1, vulcan2, vulcan3

This gets a compute node, and runs the job on it
```bash
sbatch train_letters.sh
```

To get a compute node (for dev purposes):
```bash
salloc --account=aip-lazar --gres=gpu:l40s:1 --cpus-per-task=4 --mem=16G --time=1:00:00
```