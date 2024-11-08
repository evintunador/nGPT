# Heavily simplified & edited from my templateGPT repo: https://github.com/evintunador/templateGPT
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_norm(x: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    Cosine normalization function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    # calculate the magnitude of the vectors
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-6)
    # divide by the magnitude to place on the unit hypersphere
    return x / norm

class PrecomputeRotaryFrequencies(nn.Module):
    """
    This class precomputes the RoPE frequencies based on the expected `max_seq_len` and `head_dim`.
    It uses real-valued arithmetic instead of complex numbers to ensure compatibility with MPS devices.

    Adapted from:
    https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

    Args:
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length expected.
        theta (float, optional): Base value for computing frequencies. Default is 10,000.
        device (str, optional): Device to store the frequencies on. Defaults to CUDA if available, else MPS, else CPU.
    """
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0, device: str = None):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)
        self.device = device
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=self.device).float() / head_dim))
        # Shape: (head_dim // 2)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self):
        """
        Compute the cosine and sine frequencies for rotary positional encoding.

        Returns:
            dict: A dictionary containing 'cos' and 'sin' tensors, each of shape
            (1, max_seq_len, 1, head_dim).
        """
        # Compute position indices
        t = torch.arange(self.max_seq_len, device=self.device).type_as(self.inv_freq)  # Shape: (max_seq_len)

        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # Shape: (max_seq_len, head_dim // 2)

        # Concatenate frequencies to match head_dim
        emb = torch.cat((freqs, freqs), dim=-1)  # Shape: (max_seq_len, head_dim)

        # Compute cosine and sine embeddings
        return {
            'cos': emb.cos()[None, :, None, :],  # Shape: (1, max_seq_len, 1, head_dim)
            'sin': emb.sin()[None, :, None, :]   # Shape: (1, max_seq_len, 1, head_dim)
        }


class SelfAttention(nn.Module):
    """
    A flexible self-attention module.

    Args:
        dim (int): Input and output dimension of the model.
        head_dim (int): Dimension of each attention head.
        num_heads (int): Number of heads.
        device (str, optional): Device to run the module on. 
            Defaults to CUDA if available, else MPS, else CPU.
    """
    def __init__(
        self, 
        dim: int,
        num_heads: int,
        device = None
    ):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads 

        # Define linear projections for queries, keys, and values
        self.Wq = nn.Linear(dim, num_heads * self.head_dim, bias=False, device=self.device)
        self.Wk = nn.Linear(dim, num_heads * self.head_dim, bias=False, device=self.device)
        self.Wv = nn.Linear(dim, num_heads * self.head_dim, bias=False, device=self.device)

        # the scaling factor to apply to the normalized queries & keys (see page 4)
        self.s_qk_scale = 1 / math.sqrt(dim)
        self.s_qk_init = 1. # for explanations of the scale & init, see pages 5 & 19
        self.s_qk = nn.Parameter(torch.ones(num_heads, self.head_dim, device=self.device) * self.s_qk_scale)

        # the scaling factor to apply to the attention logits to restore a variance of 1 (see page 4)
        self.scale = self.head_dim ** 0.5

        # Output projection that mixes all the attention heads back together
        self.Wo = nn.Linear(num_heads * self.head_dim, dim, bias=False, device=self.device)
        # this flag designates Wo to have a different parameter initialization as defined below in Model
        self.Wo.GPT_scale_init = 1

    def forward(self,
        x: torch.Tensor,
        freqs: dict = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for the self-attention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs (dict, optional): Precomputed rotary positional encoding frequencies.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections for queries, keys, and values
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
            # shape: (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads * head_dim)

        # Reshape projections to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # applying RoPE
        sin = freqs['sin'][:, :seq_len, :, :].to(self.device) 
        cos = freqs['cos'][:, :seq_len, :, :].to(self.device) # (1, seq_len, 1, head_dim // 2)
        q = self.apply_rotary_pos_emb(q, sin, cos) # no shape change
        k = self.apply_rotary_pos_emb(k, sin, cos)

        # normalizing & scaling our queries  & keys (see page 4)
        effective_s_qk = self.s_qk * (self.s_qk_init / self.s_qk_scale) # (head_dim)
        q = cosine_norm(q) * effective_s_qk # then scale each head
        k = cosine_norm(k) * effective_s_qk # no shape change

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2) 
        
        # Compute attention logits (compare queries & keys)
        logits = (q @ k.transpose(-2, -1)) * self.scale # (batch_size, num_heads, seq_len, seq_len)
            
        # here we mask out all the future-values
        logits = logits.masked_fill(~mask, float('-inf'))  # (batch_size, num_heads, seq_len, seq_len)

        # Compute attention scores (grab the relevant values that correspond to the attention logits)
        scores =  F.softmax(logits, dim=-1) @ v # (batch_size, n_heads, seq_len, head_dim)

        # Combine heads
        scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, -1) 
            # (batch_size, seq_len, n_heads * head_dim)
        
        return self.Wo(scores) # (batch_size, seq_len, dim)
    
    def apply_rotary_pos_emb(
        self, 
        x: torch.Tensor, 
        sin: torch.Tensor,
        cos: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary positional embeddings to queries and keys using 
        real-valued complex arithmetic. We do this because Apple silicon 
        ('mps') devices do not support complex arithmetic

        Args:
            x (torch.Tensor): Either Queries or Keys tensor.
            sin (torch.Tensor): Sine frequencies tensor.
            cos (torch.Tensor): Cosine frequencies tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
        """
        # the real component is simple
        x_real = x * cos # (batch_size, seq_len, num_heads, head_dim)
        
        # the imaginary component requires we mess with the order
        x1, x2 = x.chunk(2, dim=-1)
        x_imag = torch.cat((-x2, x1), dim=-1) * sin

        # and here are our successfully rotated q or k vectors
        x = x_real + x_imag

        return x

class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) module with optional gating and dropout.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output features.
        device (str or torch.device): Device to run the module on.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device = None
    ):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if device is None else device)

        # the up, down, and gate projections
        self.Wup = nn.Linear(input_dim, hidden_dim, bias=False, device=self.device)
        self.Wgate = nn.Linear(input_dim, hidden_dim, bias=False, device=self.device)
        self.Wdown = nn.Linear(hidden_dim, output_dim, bias=False, device=self.device)

        # this flag designates Wdown to have a different parameter initialization as defined in model.py
        self.Wdown.GPT_scale_init = 1 

        # the learnable scaling factors
        self.s_u_scale = 1. # for explanations of the scale & init, see pages 5 & 19
        self.s_u_init = 1.
        self.s_u = nn.Parameter(torch.ones(hidden_dim, device=self.device) * self.s_u_scale)
        self.s_v_scale = 1.
        self.s_v_init = 1.
        self.s_v = nn.Parameter(torch.ones(hidden_dim, device=self.device) * self.s_v_scale)

        # the varaince-controlling scaling term, needed to benefit from SiLU (see appendix A.1)
        self.scale = math.sqrt(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        # our up & gate projections
        u = self.Wup(x) # (batch_size, seq_len, hidden_dim)
        v = self.Wgate(x)
        # scale them
        effective_s_u = self.s_u * (self.s_u_init / self.s_u_scale) # (hidden_dim)
        effective_s_v = self.s_v * (self.s_v_init / self.s_v_scale)
        u = u * effective_s_u # no shape change
        v = v * effective_s_v * self.scale 
        # now perform the nonlinearity gate
        hidden = u * F.silu(v) # (batch_size, seq_len, hidden_dim)
        return self.Wdown(hidden) # (batch_size, seq_len, output_dim)
    
class Layer(nn.Module):
    """
    A single layer of the model, consisting of an attention mechanism and a feedforward connection.

    Args:
        cfg: Configuration object containing model parameters such as dimensions, number of heads, and device.

    Attributes:
        attn (SelfAttention): Self-attention mechanism.
        mlp (MLP): Multilayer Perceptron for feedforward connection.
    """
    def __init__(self, cfg):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if cfg.device is None else cfg.device)

        ### attention connection
        self.attn = SelfAttention(cfg.dim, cfg.num_heads, self.device)
        # eigen learning rate vector
        self.a_A_scale = 1. / math.sqrt(cfg.dim)
        self.a_A_init = 1. # for explanations of the scale & init, see page 5
        self.a_A = nn.Parameter(torch.ones(cfg.dim, device=self.device) * self.a_A_scale)

        ### feedforward connection
        # ensures mlp_hidden_mult maintains the same parameter count as if we were using a not-gated MLP
        mult = cfg.mlp_hidden_mult * 2/3
        self.mlp = MLP(cfg.dim, int(cfg.dim * mult),  cfg.dim, self.device)
        # eigen learning rate vector
        self.a_M_scale = 1. / math.sqrt(cfg.dim)
        self.a_M_init = 1. # for explanations of the scale & init, see page 5
        self.a_M = nn.Parameter(torch.ones(cfg.dim, device=self.device) * self.a_M_scale)

    def forward(self, h: torch.Tensor, freqs: dict, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Layer module.

        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs (dict, optional): Dictionary containing 'cos' and 'sin' tensors for rotary positional encoding.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        h_A = cosine_norm(self.attn(h, freqs, mask))
        h = cosine_norm(h + self.a_A * (h_A - h))
        h_M = cosine_norm(self.mlp(h))
        h = cosine_norm(h + self.a_M * (h_M - h))
        return h

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = (('cuda' if torch.cuda.is_available() else
                        'mps' if torch.backends.mps.is_available() else 'cpu')
                        if cfg.device is None else cfg.device)
        self.dim = cfg.dim
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len

        ### positional encodings
        self.precompute_freqs = PrecomputeRotaryFrequencies(cfg.dim // cfg.num_heads, cfg.max_seq_len, cfg.theta, self.device)
        
        # residual state initialization
        self.token_embedder = nn.Embedding(self.vocab_len, cfg.dim, device=self.device)

        # the causal attention mask
        self.mask = torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool, device=self.device).tril()
            # False -> "mask this token" while True -> "Let the model see this token"

        # the model itself
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_layers))

        # the output projection
        self.output = nn.Linear(cfg.dim, self.vocab_len, bias=False, device=self.device)
        # scaling param to un-limit the range for the final probability distribution (see page 2)
        self.s_z_scale = 1 / math.sqrt(self.dim)
        self.s_z_init = 1. # for explanations of the scale & init, see pages 5 & 19
        self.s_z = nn.Parameter(torch.ones(self.vocab_len, device=self.device) * self.s_z_scale)

        # loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.vocab_len -1) # ignore the padding token

        # initializing params to specific distributions
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        """
        parameter initialization isn't actually important in N-GPT because of the normalization
        However we'll still initialize according to how they did in appendix A.5
        """
        # whereas GPT-2 used std = 0.02, we'll do square root of model's embedding dimension
        std = math.sqrt(self.dim) 

        if isinstance(module, (nn.Linear, nn.Parameter)):
            # specific weight matrices at the end of each layer are given smaller std 
            # originally this was done in GPT-2 to keep the residual stream small
            if hasattr(module, 'GPT_scale_init'):
                std *= (2 * self.num_layers) ** -0.5

            # carries out the actual initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # biases, if any, should instead be initialized to zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 

        # the embedding matrix doesn't count as an nn.Linear so we've gotta do it again for that
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        The token embeddings are not included
        """
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.token_embedder.weight.numel()
        return n_params

    def forward(
        self, 
        input_token_ids: torch.Tensor, 
        target_token_ids: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Our N-GPT's primary forward function that calls all the other modules
        """
        input_token_ids = input_token_ids.to(self.device)
        batch_size, seq_len = input_token_ids.shape
        if target_token_ids is not None: # training setup
            target_token_ids = target_token_ids.to(self.device)
            assert batch_size, seq_len == target_token_ids.shape
            assert seq_len == self.max_seq_len
        
        # creating our causal self-attention mask
        mask = self.mask[:seq_len, :seq_len]

        # precomputing our RoPE frequencies
        freqs = self.precompute_freqs() 
            # dict {'sin': shape (1, max_seq_len, 1, head_dim), 'cos': shape (1, max_seq_len, 1, head_dim)}
      
        # initializing the first residual state
        x = self.token_embedder(input_token_ids) # (batch_size, seq_len, dim)
        
        # run through the model's layers
        for layer in self.layers:
            x = layer(x, freqs, mask)
        
        # the final output of the model
        logits = self.output(x) # (batch_size, seq_len, vocab_len)
        # to un-limit the temperature of the final probability distribution (see page 2)
        effective_s_z = self.s_z * (self.s_z_init / self.s_z_scale) # (dim)
        scaled_logits = logits * effective_s_z
        
        loss = None
        if target_token_ids is not None: # if we're training, calculate the loss
            loss = self.criterion(
                scaled_logits.view(batch_size * seq_len, self.vocab_len),
                target_token_ids.reshape(batch_size * seq_len)
            )

        return logits, loss