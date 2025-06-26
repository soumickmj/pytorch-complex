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

class LayerNorm(Module):
    r"""
    Complex-Valued Layer Normalisation
    ----------------------------------
    Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Uses whitening transformation to ensure standard normal complex distribution
    with equal variance in both real and imaginary components.

    Extending the batch normalisation whitening definitions in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**
            - Section 6
            - https://arxiv.org/abs/2302.08286

    This code has been adapted from the PyTorch implementation of LayerNorm and from github.com/josiahwsmith10/complextorch/nn/modules/layernorm.py from the following paper:
        **Smith, Josiah W. "Complex-valued neural networks for data-driven signal processing and signal understanding." arXiv preprint arXiv:2309.07948 (2023)**
            - https://arxiv.org/abs/2309.07948
    """

    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True, 
        device=None, 
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(2, 2, *normalized_shape, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.zeros(2, *normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            self.weight.data.copy_(
                0.70710678118 * torch.eye(2).view(2, 2, *([1] * len(self.normalized_shape))) #the identity matrix is scaled by sqrt(1/2)
            )
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # Sanity check to make sure the shapes match
        assert (
            self.normalized_shape == input.shape[-len(self.normalized_shape) :]
        ), "Expected normalized_shape to match last dimensions of input shape!"

        return cF.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)