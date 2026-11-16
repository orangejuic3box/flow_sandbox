# IMPORTS
import time

import torch
import torch.nn.functional as F
from torchdyn.core import NeuralODE 

# funcs for creating the data and sampling from the dist + plot

from data import sample_cats, sample_moons, discretize, undiscretize
from discrete_process_data import create_dpd, plot_dpd, animate_discrete_histograms
from utils import plot_discrete_trajectories, animate_discrete_trajectories


class DiscreteFlowMatching:
    '''
    The model represents a rate matrix
    '''
    def __init__(self, optimizer, device, model, V, data):
        self.optimizer = optimizer
        self.device = device
        self.model = model
        # for discrete data
        self.v = V
        assert data in ["catmoons", "dpd"], "incorrect data type. must be 'catmoons' or 'dpd'."
        self.data_type = data
        self.dpd_target = None

    def preprocess_data(self, data):
        # catmoons is a continuous dataset (x,y coordinates can be floats)
        # so we must discretize into integers (bins)
        # print("in preprocess")
        if self.data_type == "catmoons":
            data = discretize(data, V=self.v)
        assert data.min() >= 0 and data.max() < self.v, (
        f"[{self.data_type}] values must be in [0, {self.v - 1}], "
        f"got min={data.min()}, max={data.max()}"
        )
        # print(self.data_type)

        data = torch.as_tensor(data)
        # print(data.shape, "Before unsqueeze")

        #if [N,], convert to be [N,D]
        if data.dim() == 1:
            data = data.unsqueeze(-1)

        # print(data.shape, "After unsqueeze")
        return data

        

    def generate_source(self, batch_size):
        '''
        Returns the data in shape (N,D) (number of points, dimension size)
        '''
        if self.data_type == "catmoons":
            #(N, 2) float64 catmoons
            source = sample_cats(batch_size) 
        
        elif self.data_type == "dpd":
            #clear former calcualted target
            self.dpd_target = None
            
            #generating new data
            processed = create_dpd(seed=None,sample_size=batch_size, plotting=False)
            
            #set target
            self.dpd_target = processed[-1]
            # print(f"setting dpd tsrget? {self.dpd_target}")
            
            #(N,) int64 dpd
            source = processed[0]
        return source

    def generate_target(self, batch_size):
        '''
        Returns the data in shape (N,D) (number of points, dimension size)
        '''
        if self.data_type == "catmoons":
            return sample_moons(batch_size)
        elif self.data_type == "dpd":
            if type(self.dpd_target) == None:
                raise ValueError("Target dist. is None. generate source first")
            return self.dpd_target


    def train(self, epochs=5000, batch_size = 256):
        #define training parameters

        #FM TRAINING LOOP
        start = time.time()
        for k in range(epochs):
            self.optimizer.zero_grad()

            #generate & format data according to dataset type
            #should be a torch tensor of shape (N,D)
            x0 = (self.preprocess_data(self.generate_source(batch_size))
                .clone()
                .detach()
                .to(device=self.device) 
            )
            
            x1 = (self.preprocess_data(self.generate_target(batch_size))
                  .clone()
                  .detach()
                  .to(device=self.device)
            )

            #generates a tensor the same shape as x0 input, and fills it with random floats from [0,1)]
            t = torch.rand(batch_size, 1, device=self.device)

            # print(f"x0 shape {x0.shape} type {x0.dtype}")
            # print(f"x1 shape {x1.shape} type {x1.dtype}")
            # print(f"t has shape {t.shape} and type {t.dtype}")

            ######DISCRETE FLOW MATCHING#########
            #in CFM: xt = tx1 + (1 − t)x0
            #in DFM: xt = xt isn't a continuous line but rather looks a corrupted DISCRETE version of this line, it is either IS x1 or IS x0 w/ some probablity t and (t-1), respectively
            
            #creates a tensor of random floats from [0,1), then using < t as a threshold, convert to booleans
            discrete_mask = torch.rand_like(x0.float()) < t.view(-1, 1)
            #xt is then a tensor that looks like x1 where it discrete_mask is True and x0 where it is False
            xt = torch.where(discrete_mask, x1, x0) #[N,D] int64
            # print("xt", xt.shape, xt.dtype)
            #########################################
            #asking the model, given the (x,y) coord at time t, where do we think the final coordinate is
            if self.data_type == "catmoons":
                xt_t_input = torch.cat([xt.float() / self.v, t], dim=-1) #[N,D+1] float32
                # print("xt t input", xt_t_input.shape, xt_t_input.dtype)
                logits = self.model(xt_t_input) #[N,V*2] float32
            
            #input for transformer should be [B, N, D]
            elif self.data_type == "dpd":
                # xt_input = (xt.float() / self.v).unsqueeze(1)
                # [N, D]   -> [1, N, D]
                y = (xt.float() / self.v).unsqueeze(0)   
                # [N, 1]   -> [1, N]: one time per token
                t_seq = t.reshape(1, -1)       
                            
                logits = self.model(t_seq, y).squeeze(0)
                        
            # logits = self.model(t, xt_input)
            # print("logits", logits.reshape(-1, self.v).shape, logits.dtype)
            # print("x1", x1.reshape(-1).shape, x1.dtype)


            #calculate loss as the how close the logit prediction was to x1
            loss = F.cross_entropy(logits.reshape(-1, self.v), x1.reshape(-1))
            loss.backward()
            self.optimizer.step()

            
            if k % (epochs//10) == 0 or k == (epochs-1): #in [0,50, 250,500, 1000, 1999]:
                end = time.time()
                print(f"Epoch:{k}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                start = end

                traj = self.sample(N=1024, return_trajectory=True) 
                traj = torch.stack(traj).cpu() #[T,N,D]
                # print("traj before plotting")
                # print(traj.shape) 

                if self.data_type == "catmoons":
                    plot_discrete_trajectories(traj, V=self.v)
                    animate_discrete_trajectories(traj, V=self.v, epoch=k, save_gif=True)
                elif self.data_type == "dpd":
                    traj = traj.squeeze(-1) #drop D dimension (==1)

                    # plot_dpd(traj, n_snapshots = 3)

                    animate_discrete_histograms(traj, V=self.v, epoch=k, target=self.dpd_target, save_gif=True)

            


    @torch.no_grad()
    def sample(self, N=1500, total_steps=99, return_trajectory=False):
        '''
        Simulates the discrete flow of starting with t0 and xt as x0,
        it jumps towards its predicted x1 in total_steps, 
        at each step we change parts of xt to be the model's predicted x1 value with probability (dt/(1-t))

        Returns either the trajectory or the final predicted x1 after being simulated through the discrete flow
        '''
        # start t=0 and xt from source and jump towards target distribution
        # xt = discretize(self.generate_source(N), V=self.v).to(self.device)
        xt = (self.preprocess_data(self.generate_source(N))
            .to(device=self.device) 
        )
        
        traj = [xt.clone()]
 
        dt = 1.0 / total_steps
        for step in range(total_steps):
            t = torch.full((N, 1), step * dt, device=self.device)

            #make x1 prediction using the current xt and t
            if self.data_type == "catmoons":
                xt_t_input = torch.cat([xt.float() / self.v, t], dim=-1)
                logits = self.model(xt_t_input).view(N, self.model.dim, self.v)
            elif self.data_type == "dpd":
                # xt_input = (xt.float() / self.v).unsqueeze(1)
                # [N, D]   -> [1, N, D]
                y = (xt.float() / self.v).unsqueeze(0)   
                # [N, 1]   -> [1, N]: one time per token
                t_seq = t.reshape(1, -1)       

                logits = self.model(t_seq, y).squeeze(0)
                logits = logits.view(N, self.model.D, self.v)
            
            x1_pred = torch.distributions.Categorical(logits=logits).sample()  # (N, D)

            # print(f"xt shape {xt.shape} type {xt.dtype}")
            # print(f"x1_pred shape {x1_pred.shape} type {x1_pred.dtype}")
 
            # each coordinate jumps with prob dt / (1 - t); this equals 1 on the last step 
            jump_prob = dt / (1.0 - t)
            discrete_mask = torch.rand_like(xt.float()) < jump_prob

            # print(f"discrete_mask has shape {discrete_mask.shape} and type {discrete_mask.dtype}")

            #put together xt as jumping to our x1 prediction if True and staying as if False
            xt = torch.where(discrete_mask, x1_pred, xt)
            traj.append(xt.clone())
 
        return traj if return_trajectory else xt


    