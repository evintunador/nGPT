import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

# imports for the tokenizer
from tiny_shakespeare_tokenizer import *
tokenizer = get_tokenizer(size = 512) # size options are 128, 256, 512 and 1024

@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 12
    n_heads: int = 4
    vocab_size: int = tokenizer.vocab_len
    ffn_dim_multiplier: float = 4.
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_batch_size: int = 24
    max_seq_len: int = 512
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    dropout_rate: float = 0.1