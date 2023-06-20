import torch

def clamp(input, min=None, max=None, *, out=None):
    real = torch.clamp(input.real, min=min if type(min) is not complex else min.real, max=max if type(max) is not complex else max.real, out=out)
    imag = torch.clamp(input.imag, min=min if type(min) is not complex else min.imag, max=max if type(max) is not complex else max.imag, out=out)
    return torch.complex(real, imag)