"""
One run of the training-measures ablation: train a fresh model on n measures, evaluate it on
the test measures, and save the metrics to its own CSV.

Same setup as the ablation cell in FLOW.ipynb. On Vulcan it runs through ablation_job.sh
(one Slurm array task per (world_seed, n) pair). It also works on its own:

    python run_ablation.py --n 5 --world_seed 0
    python run_ablation.py --n 25 --world_seed 0 --epochs 500 --out_dir results/smoke   # quick timing test
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from dpd import make_measure_specs, MeasureDataset
from transformer import DiscreteAttentionFlow
from discrete_fm import DiscreteFlowMatching

TEST_CENTERS = [30, 35, 40, 45, 50]
V = 128
DIM = 1  # dpd data is 1D


def set_seed(seed):
    torch.manual_seed(seed)   # seeds CPU, CUDA and MPS generators
    np.random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="number of training measures")
    p.add_argument("--world_seed", type=int, required=True, help="torch seed: init, batches, t draws, sampling")
    p.add_argument("--train_seed", type=int, default=10, help="which centers are picked + the training pools")
    p.add_argument("--test_seed", type=int, default=1010, help="the test seed")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_n", type=int, default=1024)
    p.add_argument("--save_gif", action="store_true",
                   help="off by default: gif paths don't include the seed, so parallel tasks would overwrite each other")
    p.add_argument("--out_dir", default="results/ablation_training_measures")
    args = p.parse_args()

    device = get_device()
    print(f"device {device}, n {args.n}, world_seed {args.world_seed}, "
          f"train_seed {args.train_seed}, test_seed {args.test_seed}", flush=True)

    # data: numpy generators, seeded by train_seed / test_seed (don't touch the torch RNG)
    spec_test = make_measure_specs(centers=TEST_CENTERS)
    dpd_test = MeasureDataset(spec_test, seed=args.test_seed)
    
    spec_train = make_measure_specs(n_measures=args.n, seed=args.train_seed)
    dpd_train = MeasureDataset(spec_train, seed=args.train_seed)
    
    print(dpd_train, flush=True)

    # everything torch-random from here on is fixed by world_seed
    set_seed(args.world_seed)

    model = DiscreteAttentionFlow(
        D=DIM,
        out_dim=DIM * V,
        num_hidden=64,
        num_layers=3,
        num_heads=1,
        num_ffn_layers=2,
        num_conditions=None,
        activation=nn.GELU,
        dropout=0.05,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    dfm = DiscreteFlowMatching(optimizer, device, model, V, "dpd", dataset=dpd_train)
    start = time.time()
    #training
    dfm.train(epochs=args.epochs, batch_size=args.batch_size, save_gif=args.save_gif, display_anim=False)
    train_minutes = (time.time() - start) / 60
    print(f"training {args.n} measures took {train_minutes:.1f} min", flush=True)

    results = dfm.evaluate(dpd_test, N=args.eval_n)
    for r in results:
        r.update(world_seed=args.world_seed, 
                train_seed=args.train_seed, 
                test_seed=args.test_seed,
                epochs=args.epochs, 
                train_minutes=round(train_minutes, 2))

    out_path = os.path.join(args.out_dir, f"world_seed{args.world_seed}",
                            f"n{args.n}_trainseed{args.train_seed}_testseed{args.test_seed}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
