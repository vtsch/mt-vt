from bunch import Bunch
import torch
import torch.nn as nn
from torch import nn
#from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

def positional_encoding(config: Bunch, inp: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    '''
    Parameters:
        config: config file
        inp: input tensor
        indices: indices of the input tensor
    Returns:
        out: input tensor with positional encoding
    '''
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