import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba_array
from matplotlib.lines import Line2D
from IPython.display import HTML, display


# THIS IS FOR CONDITIONAL FLOW MATCHING, DOES NOT WORK FOR DISCRETE FLOW MATCHING
class torch_wrapper(torch.nn.Module):
    """
    Wraps MLP model to torchdyn compatible format.
    Converts [x,y,t] (MLP) to [t,[x,y]]
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj, n = 2000):
    """Plot trajectories of some selected samples."""
    #traj contains all the points, of the source dist, target dsit, and points in the path in between
    #traj looks like [t, i, coord]
    #plot the first 2000 points in traj, so i=0 to i=1999

    plt.figure(figsize=(6, 6))
    #only the source distribution so t = 0
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="steelblue")
    #plot all the points at every time, so t = :
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    #only the target distribution so t = -1 (aka last aka 1)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="gold")

    plt.legend(["Source Distribution", "Flow", "Target Distribution"])
    plt.xticks([])
    plt.yticks([])
    plt.show()

def animate_trajectories(traj, epoch, n_samples=1000, save_gif=False):
    """Saves an animated GIF of the distribution push-forward over time."""
    num_t, total_n, _ = traj.shape
    n = min(n_samples, total_n)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(traj[0, :n, 0], traj[0, :n, 1], s=8, alpha=0.7, c="steelblue")
    title = ax.set_title("t = 0.00")
    
    # Set fixed axis limits based on all trajectory bounds
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        t_val = frame / (num_t - 1)
        scat.set_offsets(traj[frame, :n])
        title.set_text(f"FM Epoch {epoch} ($t = {t_val:.2f}$)")
        return scat, title

    anim = FuncAnimation(fig, update, frames=num_t, interval=100, blit=True)
    if save_gif:
        filename = f"gifs/flow_matching/flowmatching_epoch{epoch}.gif"
        anim.save(filename, writer="pillow", fps=30)
    # Display interactive player inline
    html = anim.to_jshtml()
    display(HTML(html))
    plt.close()

def plot_discrete_trajectories(traj, V, lo=-4.0, hi=4.0, n_snapshots=2, n=2000, jitter=True):
    """
    Plot a discrete flow as a row of snapshots instead of one overlaid cloud.
 
    traj: (T, N, D) integer tokens, e.g. torch.stack(dfm.sample(return_trajectory=True))
    V:    number of bins per coordinate (same as self.v)
    lo, hi: the grid range used by discretize
    jitter: spread points inside their bin with ONE fixed offset per point,
            so a point only moves when its token actually jumps
    """
    traj = torch.as_tensor(traj)[:, :n].cpu()
    T = traj.shape[0]
 
    # tokens -> coordinates (the offset is shared across all time steps)
    width = (hi - lo) / V
    offset = torch.rand(traj.shape[1:]) if jitter else 0.5
    pts = (lo + (traj.float() + offset) * width).numpy()

    #create a mask for coloring coordinates based off whether they've fully jumped
    moved = (traj != traj[0]).sum(dim=-1).numpy()      # (T, N): 0, 1 or 2
    palette = np.array(["gold", "olive", "steelblue"])
 
    # evenly spaced time steps, always including t = 0 and t = 1
    idxs = torch.linspace(0, T - 1, n_snapshots).round().long().tolist()
    fig, axes = plt.subplots(1, n_snapshots, figsize=(3.2 * n_snapshots, 3.4))
 
    # intermediate plots
    for ax, i in zip(axes, idxs):
        color = "gold" if i == 0 else ("steelblue" if i == T - 1 else "olive")
        ax.scatter(pts[i, :, 0], pts[i, :, 1], s=2, c=palette[moved[i]])
        ax.set_title(f"t = {i / (T - 1):.2f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        
 
    plt.tight_layout()
    plt.show()


def animate_discrete_trajectories(traj, V, epoch, lo=-4.0, hi=4.0, n_samples=1000,jitter=True, max_frames=60, interval=100, save_gif=False):
    """
    Animate a discrete flow: points jump between grid bins over time.

    traj:       (T, N, 2) integer tokens, or a list of (N, 2) token tensors,
                e.g. dfm.sample(return_trajectory=True)
    V:          number of bins per coordinate (same as self.v)
    lo, hi:     the grid range used by discretize
    jitter:     place each point at ONE fixed random spot inside its bin, so it
                only moves on screen when its token actually jumps
    max_frames: discrete samplers can take many steps; frames are subsampled
                evenly down to this many (t = 0 and t = 1 always included)
    filename:   if given, also save a GIF there
    """
    # Accept a list of per-step tensors as well as a stacked tensor
    if isinstance(traj, (list, tuple)):
        traj = torch.stack([torch.as_tensor(s) for s in traj])
    traj = torch.as_tensor(traj).cpu()
    n = min(n_samples, traj.shape[1])
    traj = traj[:, :n]
    T = traj.shape[0]

    # Tokens -> coordinates; the in-bin offset is fixed per point across time
    width = (hi - lo) / V
    offset = torch.rand(n, 2) if jitter else torch.full((n, 2), 0.5)
    pts = (lo + (traj.float() + offset) * width).numpy()           # (T, n, 2)

    # Color by how many coordinates have changed from the starting token
    moved = (traj != traj[0]).sum(dim=-1).numpy()                   # (T, n): 0, 1, 2
    palette = to_rgba_array(["gold", "olive", "steelblue"])
    colors = palette[moved]                                          # (T, n, 4)

    # Pick which time steps become frames
    frames = np.unique(np.linspace(0, T - 1, min(max_frames, T)).round().astype(int))

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(pts[0, :, 0], pts[0, :, 1], s=8, alpha=0.8, c=colors[0])

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[Line2D([], [], marker="o", ls="", color=c, label=l)
                 for c, l in zip(["gold", "olive", "steelblue"],
                                 ["No jump yet", "One coord jumped", "Both coords jumped"])],
        loc="upper right", fontsize=8,
    )
    title = ax.set_title("t = 0.00")

    def update(i):
        scat.set_offsets(pts[i])
        scat.set_facecolors(colors[i])
        # title.set_text(f"Discrete Flow Matching Epoch {epoch}($t = {t_val:.2f}$)")
        title.set_text(f"Discrete FM Epoch {epoch} ($t = {i / (T - 1):.2f}$)")
        return scat, title

    # blit=False so the title (outside the axes) redraws every frame
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    if save_gif:
        filename=f"gifs/discrete_flow_matching/discreteflowmatching_epoch{epoch}.gif"
        anim.save(filename, writer="pillow", fps=18)
    display(HTML(anim.to_jshtml()))
    plt.close(fig)