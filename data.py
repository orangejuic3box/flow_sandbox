# if running asn an interactive session, make sure the kernel is the base conda envr (refresh if you cant see)
# also make sure python interpreter is the same as the kernel envr
# %%
# -------IMPORTS-------
import numpy as np 
import torch 

from torchdyn.datasets import generate_moons

#plot imports
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap 

# __all__ = ["sample_cats", "sample_moons", "plot_distribution"]

# %%
# --- Cat Distribution ---
'''
These functions are used to create a linear space that make a cat face :3
The cat distribution creates the true distribution, based on a -4.0 to 4.0 grid
'''
### GLOBAL VARIABLES TO SET GRID SPACE
MIN = -4.0
MAX = 4.0
def sign_vec(x1, y1, x2, y2, x3, y3):
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)

def in_triangle(X, Y, p1, p2, p3):
    d1 = sign_vec(X, Y, p1[0], p1[1], p2[0], p2[1])
    d2 = sign_vec(X, Y, p2[0], p2[1], p3[0], p3[1])
    d3 = sign_vec(X, Y, p3[0], p3[1], p1[0], p1[1])
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)

def create_cat_distribution(X, Y):
    """Takes X and Y meshgrids and returns a 1/0 mask of the cat."""
    #Shifts all the coordinates up and to the left
    X += 1.7
    Y -= 1.5

    #Rotate the coordinate plane
    theta = np.radians(25)
    X = X * np.cos(theta) + Y * np.sin(theta)
    Y = -X * np.sin(theta) + Y * np.cos(theta)

    #Scale down and make cat smaller
    X /= 0.7
    Y /= 0.7

    #1. Head (Now an ellipse instead of a circle)
    rx = 2.2  # Horizontal radius (wider)
    ry = 1.4  # Vertical radius (squished)
    in_head = (X**2 / rx**2) + (Y**2 / ry**2) <= 1.0

    # Ears (Triangles)
    L_apex, L_base1, L_base2 = (-1.3, 2.5), (-1.9, 0.5), (-0.2, 1.5)
    R_apex, R_base1, R_base2 = ( 1.3, 2.5), ( 1.9, 0.5), ( 0.2, 1.5)
    in_left_ear = in_triangle(X, Y, L_apex, L_base1, L_base2)
    in_right_ear = in_triangle(X, Y, R_apex, R_base1, R_base2)

    # Features (Smaller circles to carve out)
    left_eye = ((X - (-0.8))**2 + (Y - 0.4)**2) <= 0.3**2
    right_eye = ((X - 0.8)**2 + (Y - 0.4)**2) <= 0.3**2
    nose = (X**2 + (Y - (-0.4))**2) <= 0.2**2

    # Assemble the distribution mask
    cat_mask = np.zeros_like(X, dtype=int) #make array of shape X
    cat_mask[in_head | in_left_ear | in_right_ear] = 1
    cat_mask[left_eye | right_eye | nose] = 0

    #cat mask is numpy array where 0 is not cat, and 1 is
    return cat_mask


########## SAMPLE FROM CATS ####################################
def sample_cats(N=1500):
  '''
  Samples N points from the cat distribution. Returns a numpy array of those points
  
  Data sits on -4 to 4.0 grid in the X-Y axis space
  '''
  #define the xy boundaries
  x_min, x_max = MIN, MAX
  y_min, y_max = MIN, MAX
  resolution = 100  # Higher resolution means more points on the grid and smoother mask

  # 1.1 Create the mathematical grid
  # linspace creates an array with n_resolution items, spaced evenly apart between x_min and x_max
  x = np.linspace(x_min, x_max, resolution)
  y = np.linspace(y_min, y_max, resolution)
  # meshgrid makes two matrices, one of all the X coordinates, and one of all the y coordinates
  X, Y = np.meshgrid(x, y)

  # 1.2 Generate the cat array based on the specs of the meshgrid
  cat_distribution = create_cat_distribution(X, Y)


  #SAMPLE FROM CAT DISTRIBUTION
  sampled_points = []

  while len(sampled_points) < N:
      # 2.1 Randomly sample a point within the grid bounds
      x_sample = np.random.uniform(x_min, x_max)
      y_sample = np.random.uniform(y_min, y_max)

      # 2.2 Determine the closest grid index
      # (Wrapped in min() to prevent IndexError if a sample lands exactly on x_max/y_max)
      i = min(int((x_sample - x_min) / (x_max - x_min) * resolution), resolution - 1)
      j = min(int((y_sample - y_min) / (y_max - y_min) * resolution), resolution - 1)

      # 2.3 Only add the sampled point if it's in the cat distribution (aka is 1)
      if cat_distribution[j, i] == 1:
          sampled_points.append((x_sample, y_sample))

  # Convert to numpy array for easy plotting
  sampled_points = np.array(sampled_points)
  return sampled_points

########### SAMPLE FROM MOONS ###################################
def sample_moons(N=1500):
  #generate_moons from torchdyn generates N points in the form of two half moons
  x, _ = generate_moons(N, noise=0.2) #x is the collection of points
  return x + torch.tensor([1.3, -2.0]) #apply a translation to the right and down

############ PLOTS SAMPLED DISTRIBUTION #########################
def plot_distribution(sampled_points, dist_name, color):
    x_min, x_max = -4.0, 4.0
    y_min, y_max = -4.0, 4.0

    #MAKE PLOTS
    plt.figure(figsize=(6, 6))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    #Plot where the cat distribution is located on the grid space
    # plt.imshow(cat_distribution, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap=ListedColormap(["floralwhite", "cadetblue"]))
    # Plot the sampled coordinates
    names = f"{dist_name[0]}"
    for points, name, color in zip(sampled_points, dist_name, color):
      plt.scatter(points[:, 0], points[:, 1], color=color,s=4, alpha=0.7)
      if name not in names:
        names += (f" and {name}")
    plt.title(f"Sampled {len(sampled_points[0])} Points from {names} Distribution")
    plt.show()

# V is the total number of discrete bins available per coordinate dimension.
# lo and hi define the grid boundaries of the domain. set to be the MIN, MAX (-4.0 and 4.0)

def discretize(points, V, lo=MIN, hi=MAX):
    '''
    convverts points into a torch tensor and bins it according to V
    '''
    points = torch.as_tensor(points, dtype=torch.float32)
    idx = ((points - lo) / (hi - lo) * V).floor().long()
    return idx.clamp(0, V - 1)          # (N, 2) integer tokens in [0, V)

def undiscretize(idx, V, lo=MIN, hi=MAX):
    return lo + (idx.float() + 0.5) * (hi - lo) / V   # bin centers, for plotting

# # %%
# ############ TESTING ############
# #sample and plot cats
# sampled_cats = sample_cats(1500)
# plot_distribution([sampled_cats], ["Cats"], ["gold"])
# #sample and plot moons
# sampled_moons = sample_moons(1500)
# plot_distribution([sampled_moons], ["Moon"], ["steelblue"])
# #plot both together
# plot_distribution([sampled_moons, sampled_cats], ["Moon", "Cats"], ["steelblue", "gold"])