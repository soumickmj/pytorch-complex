from typing import Union, List, Tuple

import sys
import numbers
import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from .. import functional as cF
from torch.nn import init

class GroupNorm(Module):
    r"""
    Complex-Valued Group Normalisation
    ----------------------------------

    Applies Group Normalisation over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    Uses whitening transformation to ensure standard normal complex distribution
    with equal variance in both real and imaginary components.

    Extending the batch normalisation whitening definitions in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**
            - Section 6
            - https://arxiv.org/abs/2302.08286

    """
    
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int, 
        num_channels: int, 
        eps: float = 1e-5, 
        affine: bool = True,
        device=None, 
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = Parameter(torch.ones(2, 2, num_channels, **factory_kwargs))
            self.bias = Parameter(torch.zeros(2, num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            self.weight.data.copy_(
                0.70710678118 * torch.eye(2).unsqueeze(2).repeat(1, 1, self.num_channels) #the identity matrix is scaled by sqrt(1/2)
            )
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return cF.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)