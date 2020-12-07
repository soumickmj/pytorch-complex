import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter#, UninitializedParameter
from torch.nn import ParameterList
from torch.nn import functional as F
from .. import functional as cF
from torch.nn import init
from torch.nn.modules import Module
# from torch.nn.modules.lazy import LazyModuleMixin
from typing import Optional, List, Tuple, Union

from torch.nn import Identity #just to use the torch's version of identity, to help with imports

class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    complex_weights: bool
    weight: Union[Tensor, Tuple[Tensor, Tensor]]
    bias: Optional[Union[Tensor, Tuple[Tensor, Tensor]]]

    def __init__(self, in_features: int, out_features: int, bias: bool = True, complex_weights: bool = False) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.complex_weights = complex_weights

        if complex_weights:
            self.weight = Parameter(torch.Tensor(out_features, in_features).to(torch.cfloat))
        else:
            weight_real = Parameter(torch.Tensor(out_features, in_features))
            weight_imag = Parameter(torch.Tensor(out_features, in_features))
            self.weight = ParameterList([weight_real, weight_imag])

        if bias:
            if complex_weights:
                self.bias = Parameter(torch.Tensor(out_features).to(torch.cfloat))
            else:
                bias_real = Parameter(torch.Tensor(out_features))
                bias_imag = Parameter(torch.Tensor(out_features))
                self.bias = ParameterList([bias_real, bias_imag])
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _reset_parameters(self, weight, bias) -> None:
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def reset_parameters(self) -> None:
        if type(self.weight) is ParameterList:
            self._reset_parameters(self.weight[0], None if self.bias is None else self.bias[0])
            self._reset_parameters(self.weight[1], None if self.bias is None else self.bias[1])
        else:
            self._reset_parameters(self.weight, self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return cF.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript
class _LinearWithBias(Linear):
    bias: Tensor  # type: ignore

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)  # type: ignore


class Bilinear(Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1^T A x_2 + b`

    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
          :math:`*` means any number of additional dimensions. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = nn.Bilinear(20, 30, 40)
        >>> input1 = torch.randn(128, 20)
        >>> input2 = torch.randn(128, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40])
    """
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    complex_weights: bool
    weight: Union[Tensor, Tuple[Tensor, Tensor]]
    bias: Optional[Union[Tensor, Tuple[Tensor, Tensor]]]

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, complex_weights: bool = False) -> None:
        super(Bilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.complex_weights = complex_weights

        if complex_weights:
            self.weight = Parameter(torch.Tensor(out_features, in1_features, in2_features).to(torch.cfloat))
        else:
            weight_real = Parameter(torch.Tensor(out_features, in1_features, in2_features))
            weight_imag = Parameter(torch.Tensor(out_features, in1_features, in2_features))
            self.weight = ParameterList([weight_real, weight_imag])

        if bias:
            if complex_weights:
                self.bias = Parameter(torch.Tensor(out_features).to(torch.cfloat))
            else:
                bias_real = Parameter(torch.Tensor(out_features))
                bias_imag = Parameter(torch.Tensor(out_features))
                self.bias = ParameterList([bias_real, bias_imag])
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _reset_parameters(self, weight, bias) -> None:
        bound = 1 / math.sqrt(weight.size(1))
        init.uniform_(weight, -bound, bound)
        if bias is not None:
            init.uniform_(bias, -bound, bound)

    def reset_parameters(self) -> None:
        if type(self.weight) is ParameterList:
            self._reset_parameters(self.weight[0], None if self.bias is None else self.bias[0])
            self._reset_parameters(self.weight[1], None if self.bias is None else self.bias[1])
        else:
            self._reset_parameters(self.weight, self.bias)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        return cF.bilinear(input1, input2, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None
        )