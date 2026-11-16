# IMPORTS
import time

import torch
# import torchdyn
from torchdyn.core import NeuralODE 

# funcs for creating the data and sampling from the dist + plot

from data import sample_cats, sample_moons

'''
train is still a work in progress.
'''

class CountBridges:
    def __init__(self, optimizer, device, model):
        self.optimizer = optimizer
        self.device = device
        self.model = model

    def train(self):
        #define training parameters
        batch_size = 256

        #FM TRAINING LOOP
        start = time.time()
        for k in range(20000):
            self.optimizer.zero_grad()

            x0 = torch.tensor(sample_cats(batch_size), dtype=torch.float32, device=self.device)
            x1 = sample_moons(batch_size).detach().clone().to(dtype=torch.float32, device=self.device)

            # print(f"x0 shape {x0.shape} type {x0.dtype}")
            # print(f"x1 shape {x0.shape} type {x1.dtype}")

            #generates a tensor the same shape as x0 input, and fills it with random floats from [0,1)]
            t = torch.rand(x0.shape[0], device=self.device).type_as(x0).unsqueeze(-1) #shape is (batch_size,)

            ######CONDITIONAL FLOW MATCHING#########
            #randomly choose a point from each distribution, and sample a point xt along their line, ADD NOISE TO ITS POSITION SO ITS NOT EXACT
            # tx1 + (1 − t)x0
            # print(t.shape, x0.shape)
            xt = t*x1 + (1-t)*x0 + torch.rand_like(x0) * 0.01 #sigma

            #xt has a bit noise, so the conditional vector field (aka true line x0-x1), does not fully match xt
            ut = x1 - x0
            #########################################

            #concatenate xt and t together, (add a dimension to t for concatenation), dim=-1 means adding along the columns
            #vt_input is shape (batch_size, 3) aka looks like [x,y,t]
            vt_input = torch.cat([xt, t], dim=-1)

            #run input through the model to get what it thinks should be the velocity vector
            #aka asking the model, given the (x,y) coord at time t, what is the speed and direction (velocity vector) it should move in
            vt = self.model(vt_input)

            #calculate loss against "true label vector" ut
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            self.optimizer.step()

            if (k + 1) % 5000 == 0:
                end = time.time()
                print(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                start = end

                node = NeuralODE(torch_wrapper(self.model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

                with torch.no_grad():
                    #.trajectory is PyTorch's ODE Solver (torchdiffeq)
                    traj = node.trajectory(
                        torch.tensor(sample_cats(1024), dtype=torch.float32, device=self.device),
                        t_span=torch.linspace(0, 1, 100, device=self.device),
                    )
                    plot_trajectories(traj.cpu().numpy())