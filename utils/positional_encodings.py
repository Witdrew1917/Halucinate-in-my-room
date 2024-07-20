import torch
from math import pi

def fourier_encoding(x: torch.Tensor, N: int) -> torch.Tensor:
    '''
        Creates embedding vector using sinusoidal encoding.
        args:
            x: input vector of shape (b,d) where b is the batch size and d is
            a scalar
            N: dimension of fourier encoding

        OBS! Does not handle proper device placement of created tensor.
    '''

    out = []
    for i in range(N):
        out.append(torch.sin(2 ** i * pi * x))
        out.append(torch.cos(2 ** i * pi * x))
    
    return torch.cat(out, dim=-1)


