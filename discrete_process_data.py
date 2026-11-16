# if running asn an interactive session, make sure the kernel is the base conda envr (refresh if you cant see)
# also make sure python interpreter is the same as the kernel envr
# %%
# -------IMPORTS-------
import numpy as np 
import torch 
import os


from torchdyn.datasets import generate_moons

#plot imports
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

'''
(1) Make initial distribution
(2) Apply Poisson Proccess
(3) Target is initial after poisson
'''

# %%
# GLOBAL PARAMETERS
SAMPLE_TYPE = "symmetric"
PROCESS_TYPE = "poisson"
# --- 1D Histogram Distribution -----

def binomial_samples(rng, sample_size, min_center=4, max_center=50):
    '''
    min_center is the smallest bin value to center the data around
    max_center is the largest bin value to center the data round

    If we are generating train samples, center bins should range between 4-50
                         test samples, center bins should range between 60-80
    
    GIVES INTS, NO DISCRETIZATION NEEDED
    The Binomal Dist. is the discrete equivalent of a Normal Distribution
    '''
    mu = rng.integers(min_center, max_center)
    print(f"the center bin is {mu}")

    #mu-3 centers the data around mu, n=6 gives a range of 7, p0.5 make it symmetric
    samples = mu - 3 + rng.binomial(n=6, p=0.5, size=sample_size)
    
    return samples

def symmetric_samples(rng, sample_size, min_center=4, max_center=50):
    '''
    min_center is the smallest bin value to center the data around
    max_center is the largest bin value to center the data round

    If we are generating train samples, center bins should range between 4-50
                         test samples, center bins should range between 60-80
    
    GIVES INTS, NO DISCRETIZATION NEEDED            
    ''' 
    # mu = rng.integers(min_center, max_center)
    mu = 20
    # print(f"the center bin is {mu}")

    # generates data centered around 0 not 3
    half = rng.binomial(n=6, p=0.5, size=sample_size//2) - 3   
    # mirrors the half and adds together to create the dataset  
    deviations = np.concatenate([half, -half])
    # translates the data so that mu is the center
    samples = mu + deviations
    return samples
    

def apply_poisson_process(
    rng,
    source,
    steps,
    lam = 3.5,          # new events per bin per unit time
    dt = 1.0,           # length of each time step
):
    #generates the different increments of counts
    increments = rng.poisson(lam * dt, size=(steps, len(source)))

    # counts_over_time[t] is the state of every bin after t+1 steps
    counts_over_time = source + np.cumsum(increments, axis=0)
    #put source samples at the start
    counts_over_time = np.vstack([source, counts_over_time])

    return counts_over_time

def calculate_bins(samples):
    '''
    Given some data, calculate the bin_centers and counts for plotting
    '''
    # calculate the bins to be centered around an integer (edges are -0.5, 0.5)
    bins = np.arange(0, np.max(samples) + 2) - 0.5
    #density True is probability, density False are counts
    counts, bin_edges = np.histogram(samples, bins=bins, density=True) 
    # print(counts, "|",bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, counts


def create_dpd(
    sample_size=50,
    seed=42,
    timesteps = 9,
    sample_type=SAMPLE_TYPE,
    process_type=PROCESS_TYPE,
    plotting = True
):
    '''
    makes discrete process data (dpd)

    generates some type of source dist.
    applies some discrete process to create a target dist.
    

    returns an array of the data from source to target, [t,N] where t is the number of timesteps and N is number of samples
    '''
    assert sample_type in ["symmetric", "binomial"], "incorrect sample type. must be either 'binomial' or 'symmetric'."
    assert process_type in ["poisson"], "incorrect sample type. must be either 'poisson'."

    #set rng
    if seed is not None:
        print(f"Seed: {seed}, which means we should be in dev testing")
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        # seed = rng.bit_generator.seed_seq.entropy
        # print(f"the random seed from None is {seed}")

    # create an intial distribution to be centered around a specific bin
    # should be shape (N,) int64 regardless of sample type
    if sample_type =="binomial":
        samples = binomial_samples(rng, sample_size)
    elif sample_type == "symmetric":
        samples = symmetric_samples(rng, sample_size)

    # applies some process type to the source
    # should be an array of shape [t, N] where t is the number of timesteps and N is number of samples
    if process_type == "poisson":
        #shows every step of process from source to target
        # [T,N] int64
        processed = apply_poisson_process(rng, samples, timesteps)

    if plotting:
        plot_dpd(processed)

    return processed


def plot_dpd(processed, sample_type=SAMPLE_TYPE, process_type=PROCESS_TYPE, n_snapshots=2):

    #make colormap based on size of data (aka. timesteps)
    T = len(processed)
    colors = plt.cm.managua(np.linspace(0, 1, T))   # "plasma", "coolwarm", "rainbow", "virdis"

    # evenly spaced time steps, always including t = 0 and t = 1
    idxs = torch.linspace(0, T - 1, n_snapshots).round().long().tolist()

    # plot figs horizontal
    # fig, axes = plt.subplots(1, n_snapshots, figsize=(6.5 * n_snapshots, 4.2), layout="constrained")
    # plot figs vertical
    fig, axes = plt.subplots(n_snapshots, 1, figsize=(6.5, 4.2 * n_snapshots), layout="constrained")


    x_max = 102
    for ax, i in zip(axes, idxs):
        bin_centers, counts = calculate_bins([processed[i]])
        ax.bar(bin_centers, counts, width=0.9, color=colors[i], edgecolor='navy', linewidth = 0.3, alpha=0.7)
        ax.set_title(f"t = {i / (T - 1):.2f}")
        ax.set_xlim(-1, x_max)
        ax.set_xlabel('Number of Events ($k$)', fontsize=12)
        ax.set_ylabel('Probability / Relative Frequency', fontsize=12)
        ax.set_xticks(np.arange(0, x_max + 1,5))
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # for i in range(0,len(processed),len(processed)-1):  #first and last
    '''
    for i in range(len(processed)): #all of them
        bin_centers, counts = calculate_bins([processed[i]])
        plt.bar(bin_centers, counts, width=0.6, color=colors[i], edgecolor='navy', alpha=0.7, label=f't{i}')

    x_max = 102
    plt.xlim(-1, x_max)
    plt.xlabel('Number of Events ($k$)', fontsize=12)
    plt.ylabel('Probability / Relative Frequency', fontsize=12)
    plt.xticks(np.arange(0, x_max + 1,5))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    '''

    fig.suptitle(f'{process_type.upper()} Process applied to a {sample_type.upper()} Distribution', fontsize=14, fontweight='bold')
    
    # Display the chart
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
    target:   optional array of target samples, drawn as a black reference outline
    x_max:    right edge of the x-axis in real counts (defaults to the last bin)
    """
    traj = torch.as_tensor(traj).cpu()
    assert traj.dim() == 2, f"expected traj of shape (T, N), got {tuple(traj.shape)}"
    T, N = traj.shape

    def rel_freq(tokens):
        tokens = np.rint(np.asarray(tokens)).astype(int)   # guard against 9.9999 -> 9
        return np.bincount(tokens, minlength=V)[:V] / len(tokens)

    # Relative frequency of every bin at every step: shape (T, V)
    freqs = np.stack([rel_freq(traj[k].numpy()) for k in range(T)])
    x = np.arange(V) + offset                              # bin positions in real counts

    colors = plt.cm.managua(np.linspace(0, 1, T))
    frames = np.unique(np.linspace(0, T - 1, min(max_frames, T)).round().astype(int))

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(x, freqs[0], width=0.9, color=colors[0],
                  edgecolor="navy", linewidth=0.3, alpha=0.7)

    # Optional target outline, so you can see how close each frame is to the goal
    y_top = freqs.max()
    if target is not None:
        #4MAGS fix this cause wtf 
        target = torch.as_tensor(target).cpu().reshape(-1).numpy()
        t_freq = rel_freq(target - offset)
        s_freq = rel_freq(traj[0] - offset)
        ax.step(x, t_freq, where="mid", color="steelblue", linewidth=0.3, label="Target")
        ax.step(x, s_freq, where="mid", color="gold", linewidth=0.3, label="Source")
        ax.legend(loc="upper right", fontsize=8)
        y_top = max(y_top, t_freq.max())


    # Fixed axes so bar heights are comparable across frames
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








# %% 
# ############ TESTING ############
# for s in range(34, 35):
#     dpd = create_dpd(seed=s)
#     print(dpd[0])
#     print(dpd[-1])

