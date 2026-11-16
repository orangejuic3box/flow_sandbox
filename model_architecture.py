# -------IMPORTS-------
import torch 
from torch import nn

#This is the actual model that learns the vector field
class MLP(nn.Module):
  def __init__(self, dim, out_dim=None, w=64, time_varying=False):
    super().__init__()
    #time_varying is a flag for if we want to model time. it means that we can create a vector field along the time axis. rather than a single vector field
    #the data itself is always jsut he data with no timesteps associated
    self.time_varying = time_varying

    #if we are doing normal FM, data is cont. and outdim should be the same as dim
    if out_dim is None:
      out_dim = dim

    self.dim = dim
    self.out_dim = out_dim

    #MODEL ARCHITECTURE
    # 3 hidden SELU layers, pushes data towards mean mu of 0, variance sigma of 1
    # w determines how exact/sharp we want the learned vector field to be.
    self.net = torch.nn.Sequential(
      torch.nn.Linear(dim + (1 if time_varying else 0), w), #adds time dimension, so increase by 1
      torch.nn.SELU(),
      torch.nn.Linear(w, w),
      torch.nn.SELU(),
      torch.nn.Linear(w, w),
      torch.nn.SELU(),
      torch.nn.Linear(w, out_dim),
    )

  def forward(self, x):
    #runs the input x through the current NN, and gets the output of that
    return self.net(x)




























'''
# @dataclass(eq=False)
class ConditionalAttentionFlow(nn.Module):
    """
    Parameterizes time dependent vector field
    """
    D: int = 2
    num_hidden: int = 512
    num_layers: int = 3
    num_heads: int = 8
    num_ffn_layers: int = 4
    num_ffn_hidden: int = 512
    num_conditions: int = None
    activation: nn.Module = nn.GELU
    dropout: float = 0.1
    use_spatial_feat: bool = True
    num_spatial_samples: int = 128
    spatial_feat_scale: float = 0.01
    # > 1: one output projection per head behind the shared backbone, and forward takes the
    # head of every row (CHANTS joint: 0 = temporal legs, 1 = perturb legs). 1 = the single
    # projection, exactly as before (same modules, init order and state-dict keys).
    num_output_heads: int = 1

    def __post_init__(self):
        super().__init__()

        # --- 1. Input Projection ---
        input_dim = self.D
        if self.use_spatial_feat:
            input_dim += 2 * self.num_spatial_samples
            self.register_buffer("B", torch.randn((self.D, self.num_spatial_samples)) * self.spatial_feat_scale)
            
        self.input_projection = nn.Linear(input_dim, self.num_hidden)

        # --- 2. Condition Projection ---
        if self.num_conditions is None or self.num_conditions <= 0:
            self.num_conditions = 0

        self.cond_projection = nn.Sequential(
            nn.Linear(self.num_conditions, self.num_hidden),
            nn.SiLU(),
            nn.Linear(self.num_hidden, self.num_hidden)
        )

        self.time_embedder = TimestepEmbedder(self.num_hidden)
        
        # --- 3. Blocks ---
        self.blocks = nn.ModuleList([
           AdaLNBlock(
               self.num_hidden, 
               self.num_heads, 
               self.num_ffn_layers,
               self.num_ffn_hidden,
               dropout=self.dropout,
               activation=self.activation,
           )
           for _ in range(self.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.num_hidden, elementwise_affine=False)
        #format output based on the number of output heads
        if self.num_output_heads > 1:
            self.output_projection = nn.ModuleList(
                [nn.Linear(self.num_hidden, self.D) for _ in range(self.num_output_heads)])
            for proj in self.output_projection:
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
        else:
            self.output_projection = nn.Linear(self.num_hidden, self.D)

            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, t, y, cond=None, head=None):
        # y: [Batch, Seq_Len, D]
        # head: with num_output_heads > 1, the output head of every row ([Batch] ints, or one
        # int for all rows); ignored with a single head
        batch_size = y.shape[0]
        seq_len = y.shape[1]
        
        # 1. Spatial Features + Input Projection
        if self.use_spatial_feat:
            y_proj = 2.0 * torch.pi * y @ self.B
            y_spatial = torch.cat([torch.cos(y_proj), torch.sin(y_proj)], dim=-1)
            x = torch.cat([y, y_spatial], dim=-1)
        else:
            x = y
        
        x = self.input_projection(x) # [Batch, Seq, Hidden]

        # --- 2. Time Embedding (Scaled & Optimized) ---
        # FIX: Scale time by 1000.0 so 0->1 becomes 0->1000 for the embedder
        t = t * 1000.0

        # Efficient Shape Handling
        #print(t.shape)
        if t.numel() == 1:
            # CASE A: Inference (Scalar) -> Embed 1, Broadcast to all
            # Efficient: Run embedding once, copy result
            t_emb = self.time_embedder(t.reshape(1))     # [1, Hidden]
            t_emb = t_emb.view(1, 1, -1).expand(batch_size, seq_len, -1) 

        else:
            # CASE B: Training ([Batch, Seq] or [Batch, Seq, 1]) -> Embed All
            # Robust: Works for both "Global Time" (broadcasted) and "Mixed Time"
            t_flat = t.reshape(-1)                       # [Batch * Seq]
            t_emb = self.time_embedder(t_flat)           # [Batch * Seq, Hidden]
            t_emb = t_emb.view(batch_size, seq_len, -1)  # [Batch, Seq, Hidden]

        # --- 3. Condition Processing & Fusion ---
        if cond is not None:
            #raise ValueError("Condition expected")
            cond_emb = self.cond_projection(cond)
            if cond_emb.ndim == 2:
                cond_emb = cond_emb.unsqueeze(1).expand(-1, seq_len, -1)
            # FUSION: Summation (Standard AdaLN)
            # [Batch, Seq, Hidden] + [Batch, Seq, Hidden]
            adaLN_input = t_emb + cond_emb
        else:
            adaLN_input = t_emb
        # -------------------
        
        # 3. Transformer Pass
        for block in self.blocks:
            x = block(x, adaLN_input)

        x = self.final_norm(x)
        if self.num_output_heads == 1:
            out = self.output_projection(x)

            return out

        if head is None:
            raise ValueError(f"this model has {self.num_output_heads} output heads: pass the "
                             f"head of every row (head=...)")
        head = torch.as_tensor(head, device=x.device).reshape(-1).long()
        if head.numel() == 1:
            head = head.expand(batch_size)
        # every head on every row, then each row keeps its own: a row's loss only ever
        # reaches its own head's weights. The heads are Linear, so this is cheap.
        outs = torch.stack([proj(x) for proj in self.output_projection], dim=0)  # [H, B, S, D]
        return outs[head, torch.arange(batch_size, device=x.device)]
    
# @dataclass(eq=False)
class DiscreteAttentionFlow(ConditionalAttentionFlow):
  """
  Parameterizes a RATE MATRIX

  Setting a custom out_dim because of discrete bins
  (1) INDEPDENT BINS
      out_dim = 2 * |V|
      the x and y axis each get V bins, seperate from one another
  (2) JOINT BINS
      out_dim = |V|^2
      the x bins are depdendent on the x and y axis
      the y bins are depdedent on the x and y axis
  """
  def __init__(self, out_dim):
    out_dim: int = field(kw_only=True)   #makes it a requirement
    # self.out_dim = out_dim

  def __post_init__(self):
    #does __post_init__ according to whats in the parent class
    super().__post_init__()
    
    #these are the custom changes for DISCRETE specifically
    assert self.num_output_heads == 1, "DiscreteAttentionFlow should only have one output head"

    #just make sure the output_+projection is set to 
    self.output_projection = nn.Linear(self.num_hidden, self.out_dim)
    nn.init.zeros_(self.output_projection.weight)
    nn.init.zeros_(self.output_projection.bias)
'''