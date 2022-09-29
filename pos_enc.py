import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional, Any
import math
from torch import long, nn, Tensor
#from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

def positional_encoding(config, inp, indices):
    dropout = nn.Dropout(p=config.dropout)
    lpe = nn.Parameter(torch.empty(config.batch_size, config.emb_size))  # requires_grad automatically set to True
    nn.init.uniform_(lpe, -0.02, 0.02)
    #rope = RotaryEmbedding(dim=config.ts_length)

    if config.pos_enc == 'none':
        out = inp
    
    if config.pos_enc == "absolute_days" or "delta_days" or "age":
        inp = inp + indices
        out = dropout(inp)
    
    if config.pos_enc == "learnable_pos_enc":
        """Inputs of forward function 
        Args:
            inp: the sequence fed to the positional encoder model (required). [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        inp = inp + lpe[:inp.size(0), :]
        out = dropout(inp)

    """
    if config.pos_enc == "rotary_pos_enc":
        print("Using rotary positional encoding")
        print("inp shape", inp)
        out = rope(inp)
        print(out)
        out = dropout(out)
    """
    return out