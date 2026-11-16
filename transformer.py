import torch
import torch.nn as nn
import math

# These two live elsewhere in your project; change the module name to wherever they're defined.

class ConditionalAttentionFlow(nn.Module):
    """
    Parameterizes a time-dependent vector field.

    num_output_heads > 1: one output projection per head behind the shared backbone, and
    forward takes the head of every row (CHANTS joint: 0 = temporal legs, 1 = perturb legs).
    num_output_heads = 1: a single projection.
    """

    def __init__(
        self,
        D: int = 2,
        num_hidden: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        num_ffn_layers: int = 4,
        num_ffn_hidden: int = 512,
        num_conditions: int = None,
        activation: type = nn.GELU,
        dropout: float = 0.1,
        use_spatial_feat: bool = True,
        num_spatial_samples: int = 128,
        spatial_feat_scale: float = 0.01,
        num_output_heads: int = 1,
    ):
        super().__init__()

        # Store the settings as attributes, as the dataclass version did
        self.D = D
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_ffn_layers = num_ffn_layers
        self.num_ffn_hidden = num_ffn_hidden
        self.activation = activation
        self.dropout = dropout
        self.use_spatial_feat = use_spatial_feat
        self.num_spatial_samples = num_spatial_samples
        self.spatial_feat_scale = spatial_feat_scale
        self.num_output_heads = num_output_heads

        # --- 1. Input Projection ---
        input_dim = self.D
        if self.use_spatial_feat:
            input_dim += 2 * self.num_spatial_samples
            self.register_buffer(
                "B", torch.randn((self.D, self.num_spatial_samples)) * self.spatial_feat_scale
            )

        self.input_projection = nn.Linear(input_dim, self.num_hidden)

        # --- 2. Condition Projection ---
        if num_conditions is None or num_conditions <= 0:
            num_conditions = 0
        self.num_conditions = num_conditions

        self.cond_projection = nn.Sequential(
            nn.Linear(self.num_conditions, self.num_hidden),
            nn.SiLU(),
            nn.Linear(self.num_hidden, self.num_hidden),
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

        # --- 4. Output Projection(s) ---
        self.output_projection = self._build_output_projection(self.D)

    def _build_output_projection(self, out_dim):
        """Zero-initialized output layer(s); subclasses can call this with a different size."""
        if self.num_output_heads > 1:
            proj = nn.ModuleList(
                [nn.Linear(self.num_hidden, out_dim) for _ in range(self.num_output_heads)]
            )
            for p in proj:
                nn.init.zeros_(p.weight)
                nn.init.zeros_(p.bias)
        else:
            proj = nn.Linear(self.num_hidden, out_dim)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        return proj

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

        x = self.input_projection(x)  # [Batch, Seq, Hidden]

        # 2. Time Embedding: scale 0->1 to 0->1000 for the embedder
        t = t * 1000.0

        if t.numel() == 1:
            # Inference (scalar): embed once, broadcast to all
            t_emb = self.time_embedder(t.reshape(1))               # [1, Hidden]
            t_emb = t_emb.view(1, 1, -1).expand(batch_size, seq_len, -1)
        else:
            # Training ([Batch, Seq] or [Batch, Seq, 1]): embed all
            t_flat = t.reshape(-1)                                 # [Batch * Seq]
            t_emb = self.time_embedder(t_flat)                     # [Batch * Seq, Hidden]
            t_emb = t_emb.view(batch_size, seq_len, -1)            # [Batch, Seq, Hidden]

        # 3. Condition Processing & Fusion (summation, standard AdaLN)
        if cond is not None:
            cond_emb = self.cond_projection(cond)
            if cond_emb.ndim == 2:
                cond_emb = cond_emb.unsqueeze(1).expand(-1, seq_len, -1)
            adaLN_input = t_emb + cond_emb
        else:
            adaLN_input = t_emb

        # 4. Transformer Pass
        for block in self.blocks:
            x = block(x, adaLN_input)

        x = self.final_norm(x)

        if self.num_output_heads == 1:
            return self.output_projection(x)

        if head is None:
            raise ValueError(
                f"this model has {self.num_output_heads} output heads: pass the "
                f"head of every row (head=...)"
            )
        head = torch.as_tensor(head, device=x.device).reshape(-1).long()
        if head.numel() == 1:
            head = head.expand(batch_size)
        # Every head on every row, then each row keeps its own, so a row's loss only
        # reaches its own head's weights. The heads are Linear, so this is cheap.
        outs = torch.stack([proj(x) for proj in self.output_projection], dim=0)  # [H, B, S, D]
        return outs[head, torch.arange(batch_size, device=x.device)]


class DiscreteAttentionFlow(ConditionalAttentionFlow):
    """
    Parameterizes a RATE MATRIX, with a custom output size for discrete bins.

    (1) INDEPENDENT BINS: out_dim = D * V
        each axis gets its own V bins, separate from the others
    (2) JOINT BINS: out_dim = V ** D
        one bin for every combination of values across the axes
    """

    def __init__(self, out_dim: int, **kwargs):
        num_output_heads = kwargs.get("num_output_heads", 1)
        assert num_output_heads == 1, "DiscreteAttentionFlow should only have one output head"

        super().__init__(**kwargs)
        self.out_dim = out_dim

        # Replace the D-sized output layer with one sized for the discrete bins
        self.output_projection = self._build_output_projection(self.out_dim)
        print(self.out_dim, "outdim")


class AdaLNBlock(nn.Module):
    def __init__(
            self, 
            hidden_dim, 
            num_heads, 
            num_ffn_layers, 
            num_ffn_hidden=None, 
            dropout=0.1, 
            alpha=None,
            activation=nn.GELU,
            ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # FFN Construction (kept your logic)
        ffn_layers = []
        in_dim = hidden_dim
        h_dim = num_ffn_hidden if num_ffn_hidden is not None else 4 * hidden_dim
        for _ in range(num_ffn_layers - 1):
            ffn_layers.extend([
                nn.Linear(in_dim, h_dim),
                activation(),
                nn.Dropout(dropout)
            ])
            in_dim = h_dim
            
        ffn_layers.extend([
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(dropout)
        ])
        self.ffn = nn.Sequential(*ffn_layers)
        
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        
        with torch.no_grad():
            self.adaLN_modulation[1].weight.zero_()
            self.adaLN_modulation[1].bias.zero_()

    def forward(self, x, t_emb):
        # Modulation: Chunk into 6 parts
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        
        # Attention
        x_norm1 = self.norm1(x) * (1 + scale_msa) + shift_msa

        #temperature = 5.0 # Try 2.0, 5.0, or 10.0 to force "Averaging"
        #query_scaled = x_norm1 / temperature # Makes dot products smaller -> Softmax flatter

        attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + gate_msa * attn_out 

        # FFN
        x_norm2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        ffn_out = self.ffn(x_norm2)
        x = x + gate_mlp * ffn_out

        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, freq_embedding_size=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(freq_embedding_size, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.freq_embedding_size = freq_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000): #o.g. max_period=10000
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.freq_embedding_size)
        return self.mlp(t_freq)