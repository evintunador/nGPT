# Heavily edited from my templateGPT repo: https://github.com/evintunador/templateGPT
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

### TODO:
# - [ ] normalize all weights after every single training step, i think from outside the model itself
# - [ ] clamp a_A and a_A to be positive. technically this doesn't need to be done but they did it
# - [ ] adjust the learning rate of all scaling parameters, or make a note of why i'm too lazy
# - [ ] find that function i wrote awhile ago of dot-products bw vectors & compare all token embedding vectors
# - [ ] use a lower learning rate for N-GPT (see appendix A.6)

def cosine_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Cosine normalization function.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    # calculate the magnitude of the vectors
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    # divide by the magnitude to place on the unit hypersphere
    return x / norm

class PrecomputeRotaryFrequencies(nn.module):
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
        if device is None:
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
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


class SelfAttention(nn.module):
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
        head_dim: int,
        num_heads: int,
        device = None
    ):
        super().__init__()
        if device is None:
            self.device = ('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
        self.num_heads = num_heads
        self.head_dim = dim // num_heads if head_dim is None else head_dim

        # Define linear projections for queries, keys, and values
        self.Wq = nn.Linear(dim, num_heads * head_dim, bias=False, device=self.device)
        self.Wk = nn.Linear(dim, num_heads * head_dim, bias=False, device=self.device)
        self.Wv = nn.Linear(dim, num_heads * head_dim, bias=False, device=self.device)

        # the scaling factor to apply to the normalized queries & keys (see page 4)
        self.s_qk = nn.Parameter(num_heads, head_dim, device=self.device)

        # the scaling factor to apply to the attention logits to restore a variance of 1 (see page 4)
        self.scale = head_dim ** 0.5

        # Output projection that mixes all the attention heads back together
        self.Wo = nn.Linear(num_heads * head_dim, dim, bias=False, device=self.device)

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
        q, k = self.apply_rotary_pos_emb(q, k, sin, cos) # no shape change

        # normalizing our queries & keys since their values are bounded to [-1,1] but their norms are unbounded
        q = cosine_norm(q) * self.s_qk # then scale each head
        k = cosine_norm(k) * self.s_qk # no shape change

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
        q: torch.Tensor, 
        k: torch.Tensor, 
        sin: torch.Tensor,
        cos: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Apply rotary positional embeddings to queries and keys using 
        real-valued complex arithmetic. We do this because Apple silicon 
        ('mps') devices do not support complex arithmetic

        Args:
            q (torch.Tensor): Queries tensor.
            k (torch.Tensor): Keys tensor.
            sin (torch.Tensor): Sine frequencies tensor.
            cos (torch.Tensor): Cosine frequencies tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
        """
        # the real component is simple
        q_real = q * cos # (batch_size, seq_len, num_heads, head_dim)
        k_real = k * cos # (batch_size, seq_len, num_heads, head_dim)
        
        # the imaginary component requires we mess with the order (hence rotate_half)
        q1, q2 = q.chunk(2, dim=-1)
        q_imag = torch.cat((-q2, q1), dim=-1) * sin
        k1, k2 = k.chunk(2, dim=-1)
        k_imag = torch.cat((-k2, k1), dim=-1) * sin

        # and here are our successfully rotates q&k vectors
        q = q_real + q_imag
        k = k_real + k_imag

        return q, k

class MLP(nn.module):
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
        if device is None:
            device = ('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

        # the up, down, and gate projections
        self.Wup = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.Wgate = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.Wdown = nn.Linear(hidden_dim, output_dim, bias=False, device=device)

        # the learnable scaling factors
        self.s_u = nn.Parameter(hidden_dim, 1, bias=False, device=device)
        self.s_v = nn.Parameter(hidden_dim, 1, bias=False, device=device)

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
        u = u * self.s_u # no shape change
        v = v * self.s_v * self.scale
        # now perform the nonlinearity gate
        hidden = u * nn.SiLU(v) # (batch_size, seq_len, hidden_dim)
        return self.Wdown(hidden) # (batch_size, seq_len, output_dim)
    
class Layer(nn.module):
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

        ### attention connection
        self.attn = SelfAttention(cfg.dim, cfg.head_dim, cfg.num_heads, cfg.device)
        # eigen learning rate vector
        self.a_A = nn.Linear(cfg.dim, 1, bias=False, device=device)

        ### feedforward connection
        # ensures mlp_hidden_mult maintains the same parameter count as if we were using a not-gated MLP
        mult = cfg.mlp_hidden_mult * 2/3
        self.mlp = MLP(cfg.dim, int(cfg.dim * mult),  cfg.dim, cfg.device)
        # eigen learning rate vector
        self.a_M = nn.Linear(cfg.dim, 1, bias=False, device=device)

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

class Model(nn.module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device
        self.dim = cfg.dim
        self.num_layers = cfg.num_layers
        self.max_seq_len = cfg.max_seq_len
        self.vocab_len = cfg.vocab_len

        ### positional encodings
        self.precompute_freqs = PrecomputeRotaryFrequencies(cfg.head_dim, cfg.max_seq_len, cfg.theta, cfg.device)
        
        # residual state initialization
        self.token_embedder = nn.Embedding(self.vocab_len, cfg.dim, device=cfg.device)

        # the causal attention mask
        self.mask = torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool, device=cfg.device).tril()
            # False -> "mask this token" while True -> "Let the model see this token"

        # the model itself
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.num_layers))

        # the output projection
        self.output = nn.Linear(cfg.dim, self.vocab_len, bias=False, device=cfg.device)
        # scaling parameter for the logits to un-limit the range for the final probability distribution
        self.s_z = nn.Parameter(self.vocab_len, 1, bias=False, device=cfg.device) # (see page 2)

        # optionally making the output linear layer tie weights to the input embedding matrix
        self.out_weight_share = cfg.out_weight_share
        if cfg.out_weight_share: self.token_embedder.weight = self.output.weight

        # loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index = cfg.vocab_len -1) # ignore the padding token

        # initializing params to specific distributions. self.apply() applies the function to all parts of the model
        self.apply(self.__init__weights)
            # should i make this optional? not sure if this distribution is still used for modern models or just GPT2

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

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        The token embeddings get subtracted unless weight tying to the output layer is enabled
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding & (self.out_weight_share == False):
            n_params -= self.transformer.wpe.weight.numel()
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
        scaled_logits = logits * self.s_z
        
        loss = None
        if target_token_ids is not None: # if we're training
            loss = self.criterion(
                logits.view(batch_size * seq_len, self.vocab_len),
                target_token_ids.reshape(batch_size * seq_len)
            )

        return logits, loss