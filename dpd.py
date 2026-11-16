"""
Discrete process data (dpd): unpaired source/target count distributions as a PyTorch Dataset.

(1) For each MEASURE, make a source distribution symmetric around a FIXED center bin and
    apply a counting process (e.g. Poisson) to it; the result is that measure's target
(2) The DataLoader samples the source and the target of each measure INDEPENDENTLY
(3) Several measures are stacked: data has shape [N_measures, N_samples, D]

    specs  = make_measure_specs(centers=[10, 20, 30, 40, 50])
    ds     = MeasureDataset(specs, n_samples=256, seed=0)
    loader = make_dataloader(ds)               # batch_size defaults to every measure
    batch  = next(iter(loader))                # x0, x1: [5, 256, 1], measure: [5]
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display


# ======================================================================================
# 1. SOURCES: (rng, sample_size, center, half_width) -> (N,) int64, centered on `center`
# ======================================================================================
def symmetric_samples(rng, sample_size, center=20, half_width=3):
    """
    EXACTLY symmetric around the given `center`, range is [center - half_width, center + half_width]
    Binomial deviations are mirrored so every +d has a matching -d.

    If we are generating train samples, center bins should range between 4-50
                         test samples, center bins should range between 60-80
    
    GIVES INTS, NO DISCRETIZATION NEEDED
    """
    #creates one half of the dataset then mirrors it and adds together
    half = rng.binomial(n=2 * half_width, p=0.5, size=sample_size // 2) - half_width
    deviations = np.concatenate([half, -half])

    # odd size: the extra sample sits on the center
    if sample_size % 2:                       
        deviations = np.append(deviations, 0)

    # everyday we're shuffling
    rng.shuffle(deviations)
    return (center + deviations).astype(np.int64)


def binomial_samples(rng, sample_size, center=20, half_width=3):
    """
    ROUGHLY symmetric around the given `center` (plain binomial draws, symmetric on average).

    GIVES INTS, NO DISCRETIZATION NEEDED
    The Binomal Dist. is the discrete equivalent of a Normal Distribution
    """
    return (center - half_width + rng.binomial(n=2 * half_width, p=0.5, size=sample_size)).astype(np.int64)


SOURCES = {
    "symmetric": symmetric_samples,
    "binomial": binomial_samples,
}


# ======================================================================================
# 2. PROCESSES: (rng, source, steps, **kwargs) -> (steps + 1, N) trajectory
#    row 0 is the source, the last row is the target
# ======================================================================================
def apply_poisson_process(rng, source, steps, lam=3.5, dt=1.0):
    """Each step adds Poisson(lam * dt) new events to every sample."""
    increments = rng.poisson(lam * dt, size=(steps, len(source)))
    counts_over_time = source + np.cumsum(increments, axis=0)
    return np.vstack([source, counts_over_time])


def apply_bernoulli_process(rng, source, steps, p=0.5):
    """Each step adds one event with probability p (at most one event per step)."""
    increments = (rng.random((steps, len(source))) < p).astype(np.int64)
    counts_over_time = source + np.cumsum(increments, axis=0)
    return np.vstack([source, counts_over_time])


PROCESSES = {
    "poisson": apply_poisson_process,
    "bernoulli": apply_bernoulli_process,
}


# ======================================================================================
# 3. MEASURES: one source distribution + the process that turns it into its target
# ======================================================================================
@dataclass
class MeasureSpec:
    center: int = 20
    sample_type: str = "symmetric"
    process_type: str = "poisson"
    timesteps: int = 9
    half_width: int = 3
    process_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.sample_type in SOURCES, f"sample_type must be one of {list(SOURCES)}"
        assert self.process_type in PROCESSES, f"process_type must be one of {list(PROCESSES)}"
        assert self.center - self.half_width >= 0, "counts can't be negative: need center >= half_width"

    @property
    def name(self):
        extra = "_".join(f"{k}{v}" for k, v in self.process_kwargs.items())
        return f"{self.sample_type}_c{self.center}_{self.process_type}" + (f"_{extra}" if extra else "")


TaskSpec = MeasureSpec   # old name, kept so earlier code still works


def generate_task(spec, sample_size, rng):
    """Full trajectory for one measure: (timesteps + 1, sample_size) int64."""
    source = SOURCES[spec.sample_type](rng, sample_size, center=spec.center, half_width=spec.half_width)
    return PROCESSES[spec.process_type](rng, source, spec.timesteps, **spec.process_kwargs)


def make_measure_specs(n_measures=None, centers=None, min_center=4, max_center=50, seed=None, **spec_kwargs):
    """
    Build one MeasureSpec per source center. Either pass `centers` explicitly, or pass
    `n_measures` to draw that many distinct centers from [min_center, max_center).
    Any other MeasureSpec argument (process_type, timesteps, process_kwargs, ...) applies to all.
    """
    if centers is None:
        assert n_measures is not None, "pass either n_measures or centers"
        # a list of centers were not passed, so draw n_measures centers randomly
        rng = np.random.default_rng(seed)
        centers = np.sort(rng.choice(np.arange(min_center, max_center), size=n_measures, replace=False))

    #make the data
    return [MeasureSpec(center=int(c), **spec_kwargs) for c in centers]


# ======================================================================================
# 4. DATASET + DATALOADER
# ======================================================================================
class MeasureDataset(Dataset):
    """
    Unpaired source/target data for one or more measures.

    (1) For every measure, a pool of `pool_size` source samples is generated and the process
        is applied to get a pool of target samples:
            source pool [N_measures, pool_size, D],  target pool [N_measures, pool_size, D]
    (2) Each item is ONE measure: `n_samples` drawn from its source pool and, independently,
        `n_samples` from its target pool, so no source sample is linked to a target sample.

    Item:  {"x0": [n_samples, D], "x1": [n_samples, D], "measure": ()}
    Batch: {"x0": [B, n_samples, D], "x1": [B, n_samples, D], "measure": [B]}
    With batch_size = N_measures, one batch holds every measure: [N_measures, n_samples, D].
    """

    def __init__(self, specs, n_samples=256, pool_size=10_000, seed=None, replace=False):
        specs = [s if isinstance(s, MeasureSpec) else MeasureSpec(**s) for s in specs]
        assert len(specs) > 0, "need at least one measure"
        assert replace or n_samples <= pool_size, "n_samples > pool_size needs replace=True"
        self.specs = specs
        self.n_samples = n_samples
        self.pool_size = pool_size
        self.replace = replace
        self.rng = np.random.default_rng(seed)
        self.resample()

    def resample(self):
        """(Re)generate every measure's source and target pools."""
        # [N_measures, T + 1, pool_size]: the full process, kept for plotting
        self.trajectories = np.stack([generate_task(s, self.pool_size, self.rng) for s in self.specs])
        self.source = torch.as_tensor(self.trajectories[:, 0], dtype=torch.long).unsqueeze(-1)   # [M, P, 1]
        self.target = torch.as_tensor(self.trajectories[:, -1], dtype=torch.long).unsqueeze(-1)  # [M, P, 1]

    def _draw(self, n):
        if self.replace:
            return torch.randint(self.pool_size, (n,))
        return torch.randperm(self.pool_size)[:n]

    def __len__(self):
        return self.n_measures

    def __getitem__(self, m):
        # independent indices for source and target: the data is unpaired
        return {
            "x0": self.source[m, self._draw(self.n_samples)],
            "x1": self.target[m, self._draw(self.n_samples)],
            "measure": torch.tensor(m, dtype=torch.long),
        }

    def sample(self, measures=None, n_samples=None):
        """
        Draw directly, without a DataLoader (handy for sampling and evaluation).
        Returns x0, x1 of shape [len(measures), n_samples, D].
        """
        measures = range(self.n_measures) if measures is None else measures
        n = self.n_samples if n_samples is None else n_samples
        x0 = torch.stack([self.source[m, self._draw(n)] for m in measures])
        x1 = torch.stack([self.target[m, self._draw(n)] for m in measures])
        return x0, x1

    @property
    def n_measures(self):
        return len(self.specs)

    def value_range(self):
        """Smallest and largest count across every source and target pool."""
        return int(min(self.source.min(), self.target.min())), int(max(self.source.max(), self.target.max()))

    def suggest_V(self, margin=0.1):
        """A V covering every count from 0 up, with headroom for fresh resamples."""
        _, hi = self.value_range()
        return int(np.ceil((hi + 1) * (1 + margin)))

    def plot_measure(self, m, n_snapshots=2, x_max=None):
        spec = self.specs[m]
        plot_dpd(self.trajectories[m], sample_type=spec.sample_type,
                 process_type=spec.process_type, n_snapshots=n_snapshots,
                 x_max=x_max, center=spec.center)

    def __repr__(self):
        lines = [f"{m}: {s.name}" for m, s in enumerate(self.specs)]
        return (f"MeasureDataset({self.n_measures} measures, {self.n_samples} samples each, "
                f"pool {self.pool_size})\n  " + "\n  ".join(lines))


def make_dataloader(dataset, batch_size=None, shuffle=True, drop_last=False, **kwargs):
    """batch_size defaults to every measure at once: batches of [N_measures, n_samples, D]."""
    if batch_size is None:
        batch_size = len(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs)


def infinite_batches(loader):
    """Yield batches forever, so a training loop can count steps instead of epochs."""
    while True:
        for batch in loader:
            yield batch


# ======================================================================================
# 5. Backwards-compatible single-task helper
# ======================================================================================
def create_dpd(sample_size=50, 
               seed=42, 
               timesteps=9, 
               center=20,
               sample_type="symmetric", 
               process_type="poisson", 
               plotting=True, 
               **process_kwargs
):
    """
    sample_size: number of datapoints in each distribution
    """
    rng = np.random.default_rng(seed)
    spec = TaskSpec(center=center, 
                    sample_type=sample_type, 
                    process_type=process_type,
                    timesteps=timesteps, 
                    process_kwargs=process_kwargs)
    processed = generate_task(spec, sample_size, rng)
    if plotting:
        plot_dpd(processed, sample_type=sample_type, process_type=process_type, center=center)
    return processed


# ======================================================================================
# 6. PLOTTING
# ======================================================================================
def calculate_bins(samples):
    """Bin centers and relative frequencies, with bins centered on integers."""
    bins = np.arange(0, np.max(samples) + 2) - 0.5
    counts, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, counts


def plot_dpd(processed, sample_type="symmetric", process_type="poisson", n_snapshots=2,
             x_max=None, center=None):
    T = len(processed)
    colors = plt.cm.managua(np.linspace(0, 1, T))

    # evenly spaced time steps, always including t = 0 and t = 1
    idxs = np.linspace(0, T - 1, n_snapshots).round().astype(int).tolist()

    fig, axes = plt.subplots(n_snapshots, 1, figsize=(6.5, 4.2 * n_snapshots), layout="constrained")
    axes = np.atleast_1d(axes)

    if x_max is None:
        x_max = int(np.max(processed)) + 2

    for ax, i in zip(axes, idxs):
        bin_centers, counts = calculate_bins(processed[i])
        ax.bar(bin_centers, counts, width=0.9, color=colors[i], edgecolor="navy", linewidth=0.3, alpha=0.7)
        ax.set_title(f"t = {i / (T - 1):.2f}")
        ax.set_xlim(-1, x_max)
        ax.set_xlabel("Number of Events ($k$)", fontsize=12)
        ax.set_ylabel("Probability / Relative Frequency", fontsize=12)
        ax.set_xticks(np.arange(0, x_max + 1, 5))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    center_text = f" (center {center})" if center is not None else ""
    fig.suptitle(f"{process_type.upper()} Process applied to a {sample_type.upper()} Distribution{center_text}",
                 fontsize=14, fontweight="bold")
    plt.show()


def animate_discrete_histograms(traj, V, epoch=None, offset=0, target=None,
                                max_frames=60, interval=100, x_max=None,
                                title_prefix="Discrete FM", save_gif=False,
                                filename=None):
    """
    Animate how a 1D count distribution changes over time.

    traj:     (T, N) integer tokens, e.g. torch.stack(dfm.sample(return_trajectory=True)).squeeze(-1)
    V:        number of bins (same as self.v)
    offset:   added to tokens to recover real counts (0 if you didn't shift the data)
    target:   optional array of target samples; draws target and source outlines for reference
    x_max:    right edge of the x-axis in real counts (defaults to the last bin)
    """
    traj = torch.as_tensor(traj).cpu()
    assert traj.dim() == 2, f"expected traj of shape (T, N), got {tuple(traj.shape)}"
    T, N = traj.shape

    def rel_freq(tokens):
        tokens = np.rint(np.asarray(tokens)).astype(int)
        return np.bincount(tokens, minlength=V)[:V] / len(tokens)

    freqs = np.stack([rel_freq(traj[k].numpy()) for k in range(T)])     # (T, V)
    x = np.arange(V) + offset

    colors = plt.cm.managua(np.linspace(0, 1, T))
    frames = np.unique(np.linspace(0, T - 1, min(max_frames, T)).round().astype(int))

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(x, freqs[0], width=0.9, color=colors[0],
                  edgecolor="navy", linewidth=0.3, alpha=0.7)

    y_top = freqs.max()
    if target is not None:
        target = torch.as_tensor(target).cpu().reshape(-1).numpy()
        t_freq = rel_freq(target - offset)
        s_freq = freqs[0]                                               # source = first frame
        ax.step(x, t_freq, where="mid", color="steelblue", linewidth=1.2, label="Target")
        ax.step(x, s_freq, where="mid", color="gold", linewidth=1.2, label="Source")
        ax.legend(loc="upper right", fontsize=8)
        y_top = max(y_top, t_freq.max())

    ax.set_ylim(0, 1.1 * y_top)
    ax.set_xlim(x[0] - 1, x_max if x_max is not None else x[-1] + 1)
    ax.set_xlabel("Number of Events ($k$)", fontsize=12)
    ax.set_ylabel("Relative Frequency", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    epoch_text = f" Epoch {epoch}" if epoch is not None else ""
    title = ax.set_title(f"{title_prefix}{epoch_text} ($t = 0.00$)")

    def update(i):
        for bar, h in zip(bars, freqs[i]):
            bar.set_height(h)
            bar.set_facecolor(colors[i])
        title.set_text(f"{title_prefix}{epoch_text} ($t = {i / (T - 1):.2f}$)")
        return (*bars, title)

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    if save_gif:
        if filename is None:
            filename = f"gifs/discrete_flow_matching/dpd_histogram_epoch{epoch}.gif"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        anim.save(filename, writer="pillow", fps=18)
    display(HTML(anim.to_jshtml()))
    plt.close(fig)