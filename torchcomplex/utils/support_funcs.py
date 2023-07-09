import torch

def clamp(input, min=None, max=None, *, out=None):
    real = torch.clamp(input.real, min=min if type(min) is not complex else min.real, max=max if type(max) is not complex else max.real, out=out)
    imag = torch.clamp(input.imag, min=min if type(min) is not complex else min.imag, max=max if type(max) is not complex else max.imag, out=out)
    return torch.complex(real, imag)

def complex_clamp(input, min=None, max=None):
    # convert to polar coordinates
    magnitude = torch.abs(input)
    angle = torch.angle(input)

    # clamp the magnitude
    magnitude = torch.clamp(magnitude, min=min, max=max)

    # convert back to Cartesian coordinates
    clamped_real = magnitude * torch.cos(angle)
    clamped_imag = magnitude * torch.sin(angle)

    return torch.complex(clamped_real, clamped_imag)
