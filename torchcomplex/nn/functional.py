r"""Functional interface"""
import warnings
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ParameterList
from ..utils.signaltools import resample
# from torch._C import _infer_size, _add_docstr
# from torch.nn import _reduction as _Reduction
# from torch.nn.modules import utils
# from torch.nn.modules.utils import _single, _pair, _triple, _list_with_default
# from torch.nn import grad  # noqa: F401
# from torch import _VF
# from torch._jit_internal import boolean_dispatch, List, Optional, _overload, Tuple
# from torch.overrides import has_torch_function, handle_torch_function
# from torch._torch_docs import reproducibility_notes, tf32_notes

Tensor = torch.Tensor

def complex_fcaller(funtinal_handle, *args):
    return torch.view_as_complex(torch.stack((funtinal_handle(args[0].real, *args[1:]), funtinal_handle(args[0].imag, *args[1:])),dim=-1))

# Convolutions
'''Following: https://arxiv.org/pdf/1705.09792.pdf

'''
def _fcaller(funtinal_handle, *args):
    # For Convs: 0 input, 1 weight, 2 bias, 3 stride, 4 padding, 5 dilation, 6 groups
    # For ConvTrans: 0 input, 1 weight, 2 bias, 3 stride, 4 padding, 5 output_padding, 6 groups, 7 dilation

    # As PyTorch functional API only suports computations as Real-valued data, everything is converetd as Real representation of complex
    if type(args[0]) is tuple: #only incase of bilinear
        inp1 = torch.view_as_real(args[0][0])
        inp1_r = inp1[...,0]
        inp1_i = inp1[...,1]
        inp2 = torch.view_as_real(args[0][0])
        inp2_r = inp2[...,0]
        inp2_i = inp2[...,1]
    else:
        inp = torch.view_as_real(args[0])
        inp_r = inp[...,0]
        inp_i = inp[...,1]
    if type(args[1]) is ParameterList:
        w_r = args[1][0]
        w_i = args[1][1]
        if args[2] is not None:
            b_r = args[2][0]
            b_i = args[2][1]
        else:
            b_r = None
            b_i = None        
    else:
        w = torch.view_as_real(args[1])
        w_r = w[...,0]
        w_i = w[...,1]
        if args[2] is not None:
            b = torch.view_as_real(args[2])
            b_r = b[...,0]
            b_i = b[...,1]
        else:
            b_r = None
            b_i = None
    
    # Perform complex valued convolution
    if type(args[0]) is tuple: #only incase of bilinear
        MrKr = funtinal_handle(inp1_r, inp2_r, w_r, b_r, *args[3:]) #Real Feature Maps *(conv) Real Kernels
        MiKi = funtinal_handle(inp1_i, inp2_i, w_i, b_i, *args[3:]) #Imaginary Feature Maps * Imaginary Kernels
        MrKi = funtinal_handle(inp1_r, inp2_r, w_i, b_i, *args[3:]) #Real Feature Maps * Imaginary Kernels
        MiKr = funtinal_handle(inp1_i, inp2_i, w_r, b_r, *args[3:]) #Imaginary Feature Maps * Real Kernels
    else:
        MrKr = funtinal_handle(inp_r, w_r, b_r, *args[3:]) #Real Feature Maps *(conv) Real Kernels
        MiKi = funtinal_handle(inp_i, w_i, b_i, *args[3:]) #Imaginary Feature Maps * Imaginary Kernels
        MrKi = funtinal_handle(inp_r, w_i, b_i, *args[3:]) #Real Feature Maps * Imaginary Kernels
        MiKr = funtinal_handle(inp_i, w_r, b_r, *args[3:]) #Imaginary Feature Maps * Real Kernels
    real = MrKr - MiKi
    imag = MrKi + MiKr
    out = torch.view_as_complex(torch.stack((real,imag),dim=-1))
    
    return out

#Convolutions

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    return _fcaller(F.conv1d, input, weight, bias, stride, padding, dilation, groups)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    return _fcaller(F.conv2d, input, weight, bias, stride, padding, dilation, groups)

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    return _fcaller(F.conv3d, input, weight, bias, stride, padding, dilation, groups)

def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor:
    return _fcaller(F.conv_transpose1d, input, weight, bias, stride, padding, output_padding, groups, dilation)

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor:
    return _fcaller(F.conv_transpose2d, input, weight, bias, stride, padding, output_padding, groups, dilation)

def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor:
    return _fcaller(F.conv_transpose3d, input, weight, bias, stride, padding, output_padding, groups, dilation)

#Poolings
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> Tensor:
    return complex_fcaller(F.max_pool1d, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> Tensor: 
    return complex_fcaller(F.max_pool2d, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False) -> Tensor:
    return complex_fcaller(F.max_pool3d, input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)

def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor:
    return complex_fcaller(F.avg_pool1d, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor:
    return complex_fcaller(F.avg_pool2d, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor:
    return complex_fcaller(F.avg_pool3d, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

# Linear
def linear(input, weight, bias=None):
    return _fcaller(F.linear, input, weight, bias)

def bilinear(input1, input2, weight, bias=None):
    return _fcaller(F.bilinear, (input1, input2), weight, bias)


# Batch Normalizatin
def _whiten2x2_batch_norm(tensor, training=True, running_mean=None, running_cov=None,
              momentum=0.1, nugget=1e-5):
    r"""Solve R M R = I for R and a given 2x2 matrix M = [[a, b], [c, d]].

    Source: https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py

    Arguments
    ---------
    tensor : torch.tensor
        The input data expected to be at least 3d, with shape [2, B, F, ...],
        where `B` is the batch dimension, `F` -- the channels/features,
        `...` -- the spatial dimensions (if present). The leading dimension
        `2` represents real and imaginary components (stacked).

    training : bool, default=True
        Determines whether to update running feature statistics, if they are
        provided, or use them instead of batch computed statistics. If `False`
        then `running_mean` and `running_cov` MUST be provided.

    running_mean : torch.tensor, or None
        The tensor with running mean statistics having shape [2, F]. Ignored
        if explicitly `None`.

    running_cov : torch.tensor, or None
        The tensor with running real-imaginary covariance statistics having
        shape [2, 2, F]. Ignored if explicitly `None`.

    momentum : float, default=0.1
        The weight in the exponential moving average used to keep track of the
        running feature statistics.

    nugget : float, default=1e-05
        The ridge coefficient to stabilise the estimate of the real-imaginary
        covariance.

    Details
    -------
    Using (tril) L L^T = V seems to 'favour' the first dimension (re), so
    Trabelsi et al. (2018) used explicit 2x2 root of M: such R that M = RR.

    For M = [[a, b], [c, d]] we have the following facts:
        (1) inv M = \frac1{ad - bc} [[d, -b], [-c, a]]
        (2) \sqrt{M} = \frac1{t} [[a + s, b], [c, d + s]]
            for s = \sqrt{ad - bc}, t = \sqrt{a + d + 2 s}
            det \sqrt{M} = t^{-2} (ad + s(d + a) + s^2 - bc) = s

    Therefore `inv \sqrt{M} = [[p, q], [r, s]]`, where
        [[p, q], [r, s]] = \frac1{t s} [[d + s, -b], [-c, a + s]]
    """
    # assume tensor is 2 x B x F x ...

    # tail shape for broadcasting ? x 1 x F x [*1]
    tail = 1, tensor.shape[2], *([1] * (tensor.dim() - 3))
    axes = 1, *range(3, tensor.dim())

    # 1. compute batch mean [2 x F] and center the batch
    if training:
        mean = tensor.mean(dim=axes)
        if running_mean is not None:
            running_mean += momentum * (mean.data - running_mean)

    else:
        mean = running_mean

    tensor = tensor - mean.reshape(2, *tail)

    # 2. per feature real-imaginary 2x2 covariance matrix
    if training:
        # faster than doing mul and then mean. Stabilize by a small ridge.
        var = tensor.var(dim=axes, unbiased=False) + nugget
        cov_uu, cov_vv = var[0], var[1]

        # has to mul-mean here anyway (naïve) : reduction axes shifted left.
        cov_vu = cov_uv = (tensor[0] * tensor[1]).mean([a - 1 for a in axes])
        if running_cov is not None:
            cov = torch.stack([
                cov_uu.data, cov_uv.data,
                cov_vu.data, cov_vv.data,
            ], dim=0).reshape(2, 2, -1)
            running_cov += momentum * (cov - running_cov)

    else:
        cov_uu, cov_uv = running_cov[0, 0], running_cov[0, 1]
        cov_vu, cov_vv = running_cov[1, 0], running_cov[1, 1]

    # 3. get R = [[p, q], [r, s]], with E R c c^T R^T = R M R = I
    # (unsure if intentional, but the inv-root in Trabelsi et al. (2018) uses
    # numpy `np.sqrt` instead of `K.sqrt` so grads are not passed through
    # properly, i.e. constants, [complex_standardization](bn.py#L56-57).
    sqrdet = torch.sqrt(cov_uu * cov_vv - cov_uv * cov_vu)
    # torch.det uses svd, so may yield -ve machine zero

    denom = sqrdet * torch.sqrt(cov_uu + 2 * sqrdet + cov_vv)
    p, q = (cov_vv + sqrdet) / denom, -cov_uv / denom
    r, s = -cov_vu / denom, (cov_uu + sqrdet) / denom

    # 4. apply Q to x (manually)
    out = torch.stack([
        tensor[0] * p.reshape(tail) + tensor[1] * r.reshape(tail),
        tensor[0] * q.reshape(tail) + tensor[1] * s.reshape(tail),
    ], dim=0)
    return out  # , torch.cat([p, q, r, s], dim=0).reshape(2, 2, -1)

def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5, naive=False):

    """
    Source: Source: https://github.com/ivannz/cplxmodule/blob/master/cplxmodule/nn/modules/batchnorm.py
    """
    complex_weight = not(type(weight) == torch.nn.ParameterList)
    if naive:
        real = F.batch_norm(input.real,
                            running_mean[0] if running_mean is not None else None,
                            running_var[0] if running_var is not None else None,
                            weight.real if complex_weight else weight[0], bias.real if complex_weight else bias[0], training, momentum, eps)
        imag = F.batch_norm(input.imag,
                            running_mean[1] if running_mean is not None else None,
                            running_var[1] if running_var is not None else None,
                            weight.imag if complex_weight else weight[1], bias.imag if complex_weight else bias[1], training, momentum, eps)
        return torch.view_as_complex(torch.stack((real, imag),dim=-1))
    else:
        # stack along the first axis
        x = torch.stack([input.real, input.imag], dim=0)

        # whiten and apply affine transformation
        z = _whiten2x2_batch_norm(x, training=training, running_mean=running_mean,
                    running_cov=running_var, momentum=momentum, nugget=eps)

        if weight is not None and bias is not None:
            shape = 1, x.shape[2], *([1] * (x.dim() - 3))
            weight = weight.reshape(2, 2, *shape)
            z = torch.stack([
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ], dim=0) + bias.reshape(2, *shape)

        return torch.view_as_complex(torch.stack((z[0], z[1]),dim=-1))

def inv_sqrtm2x2(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    symmetric: bool = False,
):
    r"""
    Inverse Squareroot of 2x2 Matrix
    --------------------------------
    
    This code has been adapted from the PyTorch implementation of LayerNorm and from github.com/josiahwsmith10/complextorch/nn/functional.py from the following paper:
        **Smith, Josiah W. "Complex-valued neural networks for data-driven signal processing and signal understanding." arXiv preprint arXiv:2309.07948 (2023)**
            - https://arxiv.org/abs/2309.07948

    Compute the inverse matrix square root of a 2x2 matrix: :math:`A^{-1/2}`
    Improves computation speed of batch and layer normalization compared with PyTorch matrix inversion.

    Following: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix

    Given matrix :math:`\mathbf{A}` as

    .. math::
        \mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}.

    Recall

    .. math::
        \mathbf{A}^{-1} = \frac{1}{\text{det}(\mathbf{A})} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}.

    We define two parameters

    .. math::
        \delta &\triangleq \text{det}(\mathbf{A}) = ad - bc,
        \tau &\triangleq \text{trace}(\mathbf{A}) = a + d.

    Using :math:`\delta` and :math:`\tau`, we define two parameters to establish the relationship between :math:`\mathbf{A}` and its matrix square root :math:`\mathbf{A}^{1/2}` as

    .. math::
        s \triangleq \sqrt{\delta},
        t \triangleq \sqrt{\tau + 2s}.

    The matrix square root can be expressed as

    .. math::
        \mathbf{A}^{1/2} = \frac{1}{t} \begin{bmatrix} a+s & b \\ c & d+s \end{bmatrix}.

    Hence, the inverse of the matrix square root can be defined as

    .. math::
        \mathbf{A}^{-1/2} = \frac{1}{st} \begin{bmatrix} d+s & -b \\ -c & a+s \end{bmatrix}.

    Finally, defining

    .. math::

        \mathbf{B} \triangleq \begin{bmatrix} w & x \\ y & z \end{bmatrix} \triangleq \mathbf{A}^{-1/2}.

    Hence,

    .. math::

        w &= \frac{d + s}{st},
        x &= \frac{-b}{st},
        y &= \frac{-c}{st},
        z &= \frac{a + s}{st}.
    """

    if symmetric:
        # If A is symmetric, b == c and x == y
        # Hence, we ignore c and y to save one multiplaction
        delta = a * d - b * b
        tau = a + d

        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)

        coeff = 1 / (s * t)

        w, z = coeff * (d + s), coeff * (a + s)
        x, y = -coeff * b, None
    else:
        delta = a * d - b * c
        tau = a + d

        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)

        coeff = 1 / (s * t)

        w, z = coeff * (d + s), coeff * (a + s)
        x, y = -coeff * b, -coeff * c

    return w, x, y, z

def _whiten2x2_layer_norm(
    tensor: torch.Tensor,
    normalized_shape,
    eps: float = 1e-5,
):
    r"""
    Layer Normalisation Whitening
    -----------------------------

    Performs 2x2 whitening for layer normalisation.
    
    This code has been adapted from the PyTorch implementation of LayerNorm and from github.com/josiahwsmith10/complextorch/nn/functional.py from the following paper:
        **Smith, Josiah W. "Complex-valued neural networks for data-driven signal processing and signal understanding." arXiv preprint arXiv:2309.07948 (2023)**
            - https://arxiv.org/abs/2309.07948
    """
    # assume tensor is 2 x B x F x ...
    assert tensor.dim() >= 3

    # Axes over which to compute mean and covariance
    axes = [-(i + 1) for i in range(len(normalized_shape))]

    # Compute the batch mean [2, B, 1, ...] and center the batch
    mean = tensor.clone().mean(dim=axes, keepdim=True)
    tensor -= mean

    # head shape for broadcasting
    head = mean.shape[1:]

    # Compute the batch covariance [2, 2, F]
    var = (tensor * tensor).mean(dim=axes) + eps
    v_rr, v_ii = var[0], var[1]

    v_ir = (tensor[0] * tensor[1]).mean(dim=axes)

    # Compute inverse matrix square root for ZCA whitening
    p, q, _, s = inv_sqrtm2x2(v_rr, v_ir, None, v_ii, symmetric=True)

    # Whiten the batch
    return torch.stack(
        [
            tensor[0] * p.view(head) + tensor[1] * q.view(head),
            tensor[0] * q.view(head) + tensor[1] * s.view(head),
        ],
        dim=0,
    )

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    r"""
    Complex-Valued Layer Normalisation
    ----------------------------------

    Applies complex-valued layer normalisation extending the work of
    (Trabelsi et al., 2018) for each channel across a batch of data.

    Extending the batch normalisation whitening definitions in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of Complex-Valued Neural Networks.**
            - Section 6
            - https://arxiv.org/abs/2302.08286

    This code has been adapted from the PyTorch implementation of LayerNorm and from github.com/josiahwsmith10/complextorch/nn/functional.py from the following paper:
        **Smith, Josiah W. "Complex-valued neural networks for data-driven signal processing and signal understanding." arXiv preprint arXiv:2309.07948 (2023)**
            - https://arxiv.org/abs/2309.07948
    """

    # stack along the first axis
    input = torch.stack([input.real, input.imag], dim=0)

    # whiten
    z = _whiten2x2_layer_norm(
        input,
        normalized_shape,
        eps=eps,
    )

    # apply affine transformation
    if weight is not None:
        shape = *([1] * (input.dim() - 1 - len(normalized_shape))), *normalized_shape
        weight = weight.view(2, 2, *shape)
        z = torch.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            dim=0,
        ) + bias.view(2, *shape)

    return torch.view_as_complex(torch.stack((z[0], z[1]),dim=-1))

def _whiten2x2_group_norm(tensor, num_groups, eps=1e-5):
    """
    Performs 2x2 whitening for group normalisation on complex-valued tensors.
    """
    # Check for channel dimension and divisibility by num_groups
    assert tensor.dim() >= 3
    C = tensor.size(2)
    assert C % num_groups == 0, "num_channels must be divisible by num_groups"

    group_size = C // num_groups
    tensor = tensor.view(2, tensor.shape[1], num_groups, group_size, *tensor.shape[3:])

    # Compute mean and variance within groups
    mean = tensor.mean(dim=[2, 3], keepdim=True)
    tensor -= mean

    # head shape for broadcasting
    head = mean.shape[1:]

    var = (tensor * tensor).mean(dim=[2, 3], keepdim=True) + eps

    v_rr, v_ii = var[0][0], var[1][1]
    v_ir = (tensor[0] * tensor[1]).mean(dim=[2, 3], keepdim=True)[0]

    # Compute inverse square root of the covariance matrix
    p, q, _, s = inv_sqrtm2x2(v_rr, v_ir, v_ir, v_ii)

    # Whiten the tensor within each group
    z = torch.stack([
        tensor[0] * p.view(head) + tensor[1] * q.view(head),
        tensor[0] * q.view(head) + tensor[1] * s.view(head)
    ], dim=0)

    return z.reshape(z.shape[0], z.shape[1], -1, *z.shape[4:])


def group_norm(input, num_groups, weight = None, bias = None, eps = 1e-5):
    """
    Complex-Valued Group Normalisation
    ----------------------------------
    
    Applies complex-valued group normalisation for each group across a batch of data.
    """
    input_stacked = torch.stack([input.real, input.imag], dim=0)
    z = _whiten2x2_group_norm(input_stacked, num_groups, eps)

    if weight is not None and bias is not None:
        weight = weight.view(2, 2, 1, input.size(1), *([1] * (input.dim() - 2)))
        bias = bias.view(2, 1, input.size(1), *([1] * (input.dim() - 2)))

        z = torch.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            dim=0,
        ) + bias

    return torch.view_as_complex(torch.stack((z[0], z[1]), dim=-1))

# Activations

def crelu(input: Tensor, inplace: bool = False) -> Tensor:
    '''
    Eq.(4)
    https://arxiv.org/pdf/1705.09792.pdf
    '''
    if input.is_complex():
        return torch.view_as_complex(torch.stack((F.relu(input.real), F.relu(input.imag)),dim=-1))
    else:
        return F.relu(input, inplace=inplace)

def zrelu(input: Tensor, inplace: bool = False) -> Tensor:
    '''
    Guberman ReLU:
    Nitzan Guberman. On complex valued convolutional neural networks. arXiv preprint arXiv:1602.09046, 2016
    Eq.(5)
    https://arxiv.org/pdf/1705.09792.pdf
    '''
    if input.is_complex():
        return input * ((0 < input.angle()) * (input.angle() < math.pi/2)).float()
    else:
        return F.relu(input, inplace=inplace)

def modrelu(input: Tensor, bias: Tensor, inplace: bool = False) -> Tensor:
    '''
    Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.
    Notice that |z| (z.magnitude) is always positive, so if b > 0  then |z| + b > = 0 always.
    In order to have any non-linearity effect, b must be smaller than 0 (b<0).
    Update: The implementation has been updated following: \\operatorname{ReLU}(|z|+b) \\frac{z}{|z|}
    '''
    if input.is_complex():
        z_mag = torch.abs(input)
        return F.relu(z_mag + bias) * (input / z_mag)
    else:
        return F.relu(input, inplace=inplace)

def cmodrelu(input: Tensor, threshold: int, inplace: bool = False):
    r"""Compute the Complex modulus relu of the complex tensor in re-im pair.
    As proposed in : https://arxiv.org/pdf/1802.08026.pdf
    Source: https://github.com/ivannz/cplxmodule"""
    if input.is_complex():
        modulus = torch.clamp(torch.abs(input), min=1e-5)
        _tmp_newshape = (1,len(threshold)) + (1,)*len(input.shape[2:])
        return input * F.relu(1. - threshold.view(_tmp_newshape) / modulus)
    else:
        return F.relu(input, inplace=inplace)

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    '''
    Complex-valued Neural Networks with Non-parametric Activation Functions
    (Eq. 36)
    https://arxiv.org/pdf/1802.08026.pdf
    '''
    if input.is_complex():
        return F.softmax(torch.abs(input), dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def tanh(input: Tensor):
    if input.is_complex():
        a, b = input.real, input.imag
        denominator = torch.cosh(2*a) + torch.cos(2*b)
        real = torch.sinh(2 * a) / denominator
        imag = torch.sin(2 * a) / denominator
        return torch.view_as_complex(torch.stack((real, imag),dim=-1))
    else:
        return F.tanh(input)
    
def hirose(input: Tensor, m_sqaure: float = 1):
    '''
    A. Hirose. Complex-valued neural networks: Advances and applications. John Wiley & Sons, 2013. 
    and
    Wolter and Yao. Complex Gated Recurrent Neural Networks. NeurIPS 2018. (Eq. 5) https://papers.nips.cc/paper_files/paper/2018/file/652cf38361a209088302ba2b8b7f51e0-Paper.pdf
    '''
    mag_input = torch.abs(input)
    return F.tanh(mag_input/m_sqaure) * (input / mag_input)

def modsigmoid(input: Tensor, alpha: float = 0.5):
    '''
    Wolter and Yao. Complex Gated Recurrent Neural Networks. NeurIPS 2018. (Eq. 13) https://papers.nips.cc/paper_files/paper/2018/file/652cf38361a209088302ba2b8b7f51e0-Paper.pdf
    and
    Xie et al. Complex Recurrent Variational Autoencoder with Application to Speech Enhancement. 2023. arXiv:2204.02195v2
    '''
    return torch.sigmoid(alpha * input.real + (1 - alpha) * input.imag)

def csilu(input: Tensor):
    '''
    Complex-Valued Sigmoid-Weighted Linear Unit (C-SiLU) [Developed in-house by Soumick Chatterjee, yet to be proposed in a paper]
    This function effectively scales the complex number by the sigmoid of its modulus, thereby preserving the phase but modulating the magnitude based on the sigmoid function.

     .. math::
        \text{csilu}(x) = x * \sigma(|x|), \text{where } \sigma(|x|) \text{ is the logistic sigmoid performed on the magnitude of the complex input.}
    '''
    return input * torch.sigmoid(torch.abs(input))

def cgelu(input: Tensor, mode='separate'):
    '''
    Complex-Valued Gaussian Error Linear Unit (C-GELU) 
    This can be either applied to the real and imaginary parts separately as [mode=saperate]:

     .. math::
        \text{cgelu}(x) = \operatorname{GELU}(Re(x)) + i \cdot \operatorname{GELU}(Im(x))

    or, it can be applied only to the magnitude, preserving the phase as [mode=magnitude]:

     .. math::
        \text{cgelu}(x) = \operatorname{GELU}(|x|) \cdot e^{i \cdot \arg (x)}
    '''
    if mode == 'separate':
        return complex_fcaller(F.gelu, input)   
    else:
        magnitude = torch.abs(input)
        phase = torch.angle(input)
        norm_magnitude = F.gelu(magnitude)
        norm_real_part = norm_magnitude * torch.cos(phase)
        norm_imag_part = norm_magnitude * torch.sin(phase)
        return torch.complex(norm_real_part, norm_imag_part)

def sigmoid(input: Tensor):
    if input.is_complex():
        a, b = input.real, input.imag
        denominator = 1 + 2 * torch.exp(-a) * torch.cos(b) + torch.exp(-2 * a)
        real = 1 + torch.exp(-a) * torch.cos(b) / denominator
        imag = torch.exp(-a) * torch.sin(b) / denominator
        return torch.view_as_complex(torch.stack((real, imag),dim=-1))
    else:
        return F.sigmoid(input)

def _sinc_interpolate(input, size):
    axes = np.argwhere(np.equal(input.shape[2:], size) == False).squeeze(1) #2 dims for batch and channel
    out_shape = [size[i] for i in axes]
    return resample(input, out_shape, axis=axes+2) #2 dims for batch and channel

def interpolate(input, size=None, scale_factor=None, mode='sinc', align_corners=None, recompute_scale_factor=None):  
    if mode in ('nearest', 'area', 'sinc'):
        if align_corners is not None:
            raise ValueError("align_corners option can only be set with the "
                             "interpolating modes: linear | bilinear | bicubic | trilinear")

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError('only one of size or scale_factor should be defined')
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError('size shape must match input shape. '
                                 'Input is {}D, size is {}'.format(dim, len(size)))
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError('scale_factor shape must match input shape. '
                                 'Input is {}D, scale_factor is {}'.format(dim, len(scale_factor)))
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError('either size or scale_factor should be defined')

    if recompute_scale_factor is None:
        # only warn when the scales have floating values since
        # the result for ints is the same with/without recompute_scale_factor
        if scale_factors is not None:
            for scale in scale_factors:
                if math.floor(scale) != scale:
                    warnings.warn("The default behavior for interpolate/upsample with float scale_factor changed "
                                  "in 1.6.0 to align with other frameworks/libraries, and now uses scale_factor directly, "
                                  "instead of relying on the computed output size. "
                                  "If you wish to restore the old behavior, please set recompute_scale_factor=True. "
                                  "See the documentation of nn.Upsample for details. ")
                    break
    elif recompute_scale_factor and size is not None:
        raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

    # "area" and "sinc" modes always require an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if (mode == "area" or mode == "sinc") and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [(torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i],
                           dtype=torch.float32)).float())) for i in range(dim)]
        else:
            assert scale_factors is not None
            output_size = [int(math.floor(float(input.size(i + 2)) * scale_factors[i])) for i in range(dim)]
        scale_factors = None

    if mode == "sinc":
        return _sinc_interpolate(input, output_size)
    else:
        return complex_fcaller(F.interpolate, input, output_size, scale_factors, mode, align_corners)    