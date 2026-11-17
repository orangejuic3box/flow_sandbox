# IMPORTS
import time

import torch
import torch.nn.functional as F

# funcs for creating the data and sampling from the dist + plot

from data import sample_cats, sample_moons, discretize, undiscretize
# from discrete_process_data import create_dpd, plot_dpd, animate_discrete_histograms

from dpd import make_dataloader, infinite_batches, animate_discrete_histograms

from utils import plot_discrete_trajectories, animate_discrete_trajectories


class DiscreteFlowMatching:
    '''
    The model represents a rate matrix
    '''
    def __init__(self, optimizer, device, model, V, data, dataset=None):
        self.optimizer = optimizer
        self.device = device
        self.model = model
        # for discrete data
        self.v = V
        assert data in ["catmoons", "dpd"], "incorrect data type. must be 'catmoons' or 'dpd'."
        self.data_type = data

        # add dataset and batches attributes if we are dealing with dpd data
        if data == "dpd":
            assert dataset is not None, "pass a MeasureDataset for data='dpd'"
            _, hi = dataset.value_range()
            assert hi < V, f"largest count {hi} doesn't fit in V={V}; try V={dataset.suggest_V()}"
            self.dataset = dataset
            self.dpd_target = None

    def preprocess_data(self, data):
        # catmoons is a continuous dataset (x,y coordinates can be floats)
        # so we must discretize into integers (bins)
        # print("in preprocess", self.data_type)
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

        

    def sample_source(self, batch_size):
        '''
        Returns the data in shape (N,D) (number of points, dimension size)
        '''
        if self.data_type == "catmoons":
            #(N, 2) float64 catmoons
            source = sample_cats(batch_size) 
        
        elif self.data_type == "dpd":
            #grab source and target from dataset, set dpt_target
            source, self.dpd_target = self.dataset.sample(n_samples=batch_size)
        return source

    def sample_target(self, batch_size):
        '''
        Returns the data in shape (N,D) (number of points, dimension size)
        '''
        if self.data_type == "catmoons":
            return sample_moons(batch_size)
        elif self.data_type == "dpd":
            if self.dpd_target is None:
                raise ValueError("Target dist. is None. sample source first")
            return self.dpd_target        


    def train(self, epochs=5000, batch_size = 256):
        #define training parameters
        self.model.train()

        #FM TRAINING LOOP
        start = time.time()
        for k in range(epochs):
            self.optimizer.zero_grad()

            #generate & format data according to dataset type
            #should be a torch tensor of shape (N,D)
            x0 = (self.preprocess_data(self.sample_source(batch_size))
                .clone()
                .detach()
                .to(device=self.device) 
            )
            
            x1 = (self.preprocess_data(self.sample_target(batch_size))
                  .clone()
                  .detach()
                  .to(device=self.device)
            )

            #generates a tensor the same shape as x0 input, and fills it with random floats from [0,1)]
            if self.data_type == "dpd":
                #M is n_measures, B is batch_size
                M, B, _ = x0.shape
                t = torch.rand(M, 1, 1, device=self.device)               # one time per measure
            elif self.data_type == "catmoons":
                t = torch.rand(batch_size, 1, device=self.device)

            # print(f"x0 shape {x0.shape} type {x0.dtype}")
            # print(f"x1 shape {x1.shape} type {x1.dtype}")
            # print(f"t has shape {t.shape} and type {t.dtype}")

            ######DISCRETE FLOW MATCHING#########
            #in CFM: xt = tx1 + (1 − t)x0
            #in DFM: xt = xt isn't a continuous line but rather looks a corrupted DISCRETE version of this line, it is either IS x1 or IS x0 w/ some probablity t and (t-1), respectively
            
            #creates a tensor of random floats from [0,1), then using < t as a threshold, convert to booleans
            discrete_mask = torch.rand_like(x0.float()) < t
            
            #xt is then a tensor that looks like x1 where it discrete_mask is True and x0 where it is False
            xt = torch.where(discrete_mask, x1, x0) #[N,D] int64
            # print("xt", xt.shape, xt.dtype)
            #########################################
            #asking the model, given the (x,y) coord at time t, where do we think the final coordinate is
            if self.data_type == "catmoons":
                xt_t_input = torch.cat([xt.float() / self.v, t], dim=-1) #[N,D+1] float32
                # print("xt t input", xt_t_input.shape, xt_t_input.dtype)
                logits = self.model(xt_t_input) #[N,V*2] float32
            
            #input for transformer should be [M, B, N, D]
            elif self.data_type == "dpd":    
                # [M, B, 1]
                y = xt.float() / self.v 
                # t per token, but constant within a measure; [M, B, V]
                logits = self.model(t.expand(M, B, 1), y)                 
                        
            # logits = self.model(t, xt_input)
            # print("logits", logits.reshape(-1, self.v).shape, logits.dtype)
            # print("x1", x1.reshape(-1).shape, x1.dtype)


            #calculate loss as the how close the logit prediction was to x1
            loss = F.cross_entropy(logits.reshape(-1, self.v), x1.reshape(-1))
            loss.backward()
            self.optimizer.step()

            
            if k % max(1, epochs // 10) == 0 or k == (epochs-1): #in [0,50, 250,500, 1000, 1999]:
                end = time.time()
                print(f"Epoch:{k}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                start = end

                self.model.eval()
                traj = self.sample(N=1024, return_trajectory=True) 
                self.model.train()

                traj = torch.stack(traj).cpu() #[T,N,D]
                # print("traj before plotting")
                # print(traj.shape) 

                if self.data_type == "catmoons":
                    plot_discrete_trajectories(traj, V=self.v)
                    animate_discrete_trajectories(traj, V=self.v, epoch=k, save_gif=True)
                elif self.data_type == "dpd":
                    traj = traj.squeeze(-1) #drop D dimension (==1)

                    for m, spec in enumerate(self.dataset.specs):
                        # plot_dpd(traj, n_snapshots = 3)

                        animate_discrete_histograms(
                            traj[:, m], 
                            V=self.v, 
                            epoch=k,
                            target=self.dpd_target[m, :, 0].cpu(),
                            center=spec.center, 
                            title_prefix=f"{spec.process_type.upper()} Process applied to a {spec.sample_type.upper()} Distribution (center {spec.center})"
                        )


            

    @torch.no_grad()
    def sample(self, N=1500, total_steps=99, return_trajectory=False):
        '''
        Simulates the discrete flow of starting with t0 and xt as x0,
        it jumps towards its predicted x1 in total_steps, 
        at each step we change parts of xt to be the model's predicted x1 value with probability (dt/(1-t))

        Returns either the trajectory or the final predicted x1 after being simulated through the discrete flow
        '''
        # start t=0 and xt from source and jump towards target distribution
        # xt = discretize(self.sample_source(N), V=self.v).to(self.device)
        xt = (self.preprocess_data(self.sample_source(N))
            .to(device=self.device) 
        )
        M = xt.shape[0] if self.data_type == "dpd" else None
        
        traj = [xt.clone()]
 
        dt = 1.0 / total_steps
        for step in range(total_steps):

            #make x1 prediction using the current xt and t
            if self.data_type == "catmoons":
                t = torch.full((N, 1), step * dt, device=self.device)
                xt_t_input = torch.cat([xt.float() / self.v, t], dim=-1)
                logits = self.model(xt_t_input).view(N, self.model.dim, self.v)
            elif self.data_type == "dpd":
                t = torch.full((M, N, 1), step * dt, device=self.device)
            
                logits = self.model(t, xt.float() / self.v).view(M, N, self.model.D, self.v)

 
            
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


    