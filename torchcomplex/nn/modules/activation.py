
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from .. import functional as cF
from torch.nn.modules import Module
from typing import Optional, List, Tuple, Union

class GenericComplexActivation(Module):
    def __init__(self, activation, use_phase: bool = False):
        '''
        activation can be either a function from nn.functional or an object of nn.Module if the ativation has learnable parameters
        Original idea from: https://github.com/albanD
        '''
        self.activation = activation
        self.use_phase = use_phase

    def forward(self, input: Tensor):
        if self.use_phase:
            return self.activation(torch.abs(input)) * torch.exp(1.j * torch.angle(input)) 
        else:
            return self.activation(input.real) + 1.j * self.activation(input.imag)

class CReLU(Module):
    '''
    Eq.(4)
    https://arxiv.org/pdf/1705.09792.pdf
    '''
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.crelu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class zReLU(Module):
    '''
    Guberman ReLU:
    Nitzan Guberman. On complex valued convolutional neural networks. arXiv preprint arXiv:1602.09046, 2016
    Eq.(5)
    https://arxiv.org/pdf/1705.09792.pdf
    
    Warning:
    Inplace will only be used if the input is real (i.e. while using the default relu of PyTorch)
    '''
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(zReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return cF.zrelu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class modReLU(Module):
    '''
    Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.
    Notice that |z| (z.magnitude) is always positive, so if b > 0  then |z| + b > = 0 always.
    In order to have any non-linearity effect, b must be smaller than 0 (b<0).
    
    Warning:
    Inplace will only be used if the input is real (i.e. while using the default relu of PyTorch)
    '''
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, bias: int, inplace: bool = False):
        super(modReLU, self).__init__()
        self.inplace = inplace
        self.bias = bias

    def forward(self, input: Tensor) -> Tensor:
        return cF.modrelu(input, bias=self.bias, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class CmodReLU(Module):
    '''Compute the Complex modulus relu of the complex tensor in re-im pair.
    As proposed in : https://arxiv.org/pdf/1802.08026.pdf
    
    If threshold=None then it becomes a learnable parameter.
    '''
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, threshold: int, inplace: bool = False):
        super(CmodReLU, self).__init__()
        self.inplace = inplace
        if not isinstance(threshold, float):
            threshold = Parameter(torch.rand(1) * 0.25)
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return cF.cmodrelu(input, threshold=self.threshold, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class AdaptiveCmodReLU(Module):
    '''Compute the Complex modulus relu of the complex tensor in re-im pair.
    As proposed in : https://arxiv.org/pdf/1802.08026.pdf
    
    AdaptiveCmodReLU(1) learns one common threshold for all features, AdaptiveCmodReLU(d) learns seperate ones for each dimension
    '''
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, *dim):
        super(AdaptiveCmodReLU, self).__init__()
        self.dim = dim if dim else (1,)
        self.threshold = Parameter(torch.randn(*self.dim) * 0.02)

    def forward(self, input: Tensor) -> Tensor:
        return cF.cmodrelu(input, threshold=self.threshold, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
        
class Softmax(Module):
    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super(Softmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return cF.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)

class Softmax2d(Module):
    def forward(self, input: Tensor) -> Tensor:
        assert input.dim() == 4, 'Softmax2d requires a 4D tensor as input'
        return cF.softmax(input, 1, _stacklevel=5)

class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return cF.tanh(input)

class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return cF.sigmoid(input)