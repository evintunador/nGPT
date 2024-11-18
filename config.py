from dataclasses import dataclass
import torch
import time

@dataclass
class ModelConfig:
    """
    Design your N-GPT here
    """
    dim: int = 128
    device: str = None
        # defaults to best available GPU/CPU
    max_seq_len: int = 384 # 512 is the most my 8gb of ram can handle
    theta: float = 10_000 # RoPE hyperparameter; 10_000 is the most common choice
    vocab_len: int = 2048 # options are 512, 1024, 2048
    num_layers: int = 8
    num_heads: int = 4 # number of heads in the multi-head attention mechanism
    mlp_hidden_mult: float = 4 # how wide the hidden dimension of the MLP should be
    
    def __post_init__(self):
        """
        These are just checks to make sure everything works ahead of time. do not edit them unelss you know what you're doing
        """
        assert isinstance(self.dim, int) and self.dim > 0, "dim must be a positive integer"
        assert self.device in [None, 'cuda', 'mps', 'cpu'], "device must be None, 'cuda', 'mps', or 'cpu'"
        assert isinstance(self.max_seq_len, int) and self.max_seq_len > 0, "max_seq_len must be a positive integer"
        assert self.theta > 0, "theta must be a positive number"
        assert isinstance(self.vocab_len, int) and self.vocab_len > 0, "vocab_len must be a positive integer"
        assert isinstance(self.num_layers, int) and self.num_layers > 0, "num_layers must be a positive integer"
        assert self.mlp_hidden_mult > 0, "mlp_hidden_mult must be a positive number"
        assert isinstance(self.num_heads, int) and self.num_heads > 0, "num_q_heads must be a positive integer"
        
@dataclass
class TrainConfig:
    """
    Design your training loop here
    """
    # name of the folder the model will be saved into
    model_name: str = f'N-GPT_2m'

    ### batch size hyperparams
    # micro_batch_size * grad_accum_steps = effective batch size
    # micro_batch_size * grad_accum_steps * max_seq_len = total number of tokens per batch
    micro_batch_size: int = 4
    grad_accum_steps: int = 16
        # set grad_accum_steps = 1 to not do gradient accumulation

    ### training length
    # total number of batches to run over the course of training
    max_iters: int = 1000 # we'll refer to iterations of batches instead of epochs over the dataset
    # how often to print out an update on how training is going
    eval_interval: int = 100
    
    ### AdamW Hyperparameters https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    # and N-GPT disables weight-decay in the optimizer because it would pull values off of the unit-hypersphere

    ### Learning Rate Schedule
    # Maximum and minimum learning rates during annealing
    lr_init: float = 1e-2 # N-GPT does NOT use learning rate warmup
    lr_final: float = 1e-6

    def __post_init__(self):
        """
        These are just checks to make sure everything works ahead of time. do not edit them unelss you know what you're doing
        """
        assert isinstance(self.model_name, str) and len(self.model_name) > 0, "model_name must be a non-empty string"
    
        # Batch size 
        assert isinstance(self.micro_batch_size, int) and self.micro_batch_size > 0, "micro_batch_size must be a positive integer"
        assert isinstance(self.grad_accum_steps, int) and self.grad_accum_steps > 0, "grad_accum_steps must be a positive integer"
    
        # Training length 
        assert isinstance(self.max_iters, int) and self.max_iters > 0, "max_iters must be a positive integer"
        assert isinstance(self.eval_interval, int) and self.eval_interval > 0, "eval_interval must be a positive integer"
        
        # AdamW hyperparameter 
        assert 0 < self.beta1 < 1, "beta1 must be between 0 and 1"
        assert 0 < self.beta2 < 1, "beta2 must be between 0 and 1"
        assert self.epsilon > 0, "epsilon must be a positive number"
    
        # Learning rate schedule 
        assert self.lr_init > 0, "lr_init must be a positive number"
        assert self.lr_final > 0, "lr_final must be a positive number"
        assert self.lr_final <= self.lr_init, "lr_final must be less than or equal to lr_init"
        