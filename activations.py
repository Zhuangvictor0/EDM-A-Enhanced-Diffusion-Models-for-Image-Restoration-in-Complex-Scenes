from torch import nn
import numpy as np


def get_activation(act_fn):
    if act_fn in ["swish", "silu"]:
        return nn.SiLU()
    elif act_fn == "mish":
        return nn.Mish()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "sigmoid":
        return nn.Sigmoid() 
    elif act_fn == "tanh":
        return nn.Tanh()   
    elif act_fn == "leakyrelu":
        return nn.LeakyReLU()        
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
