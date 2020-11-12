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

# Linear
def linear(input, weight, bias=None):
    return _fcaller(F.linear, input, weight, bias)

def bilinear(input1, input2, weight, bias=None):
    return _fcaller(F.bilinear, (input1, input2), weight, bias)


# Batch Normalizatin
def _whiten2x2(tensor, training=True, running_mean=None, running_cov=None,
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
    if naive:
        real = F.batch_norm(input.real,
                            running_mean[0] if running_mean is not None else None,
                            running_var[0] if running_var is not None else None,
                            weight[0], bias[0], training, momentum, eps)
        imag = F.batch_norm(input.imag,
                            running_mean[1] if running_mean is not None else None,
                            running_var[1] if running_var is not None else None,
                            weight[1], bias[1], training, momentum, eps)
        return torch.view_as_complex(torch.stack((real, imag),dim=-1))
    else:
        # stack along the first axis
        x = torch.stack([input.real, input.imag], dim=0)

        # whiten and apply affine transformation
        z = _whiten2x2(x, training=training, running_mean=running_mean,
                    running_cov=running_var, momentum=momentum, nugget=eps)

        if weight is not None and bias is not None:
            shape = 1, x.shape[2], *([1] * (x.dim() - 3))
            weight = weight.reshape(2, 2, *shape)
            z = torch.stack([
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ], dim=0) + bias.reshape(2, *shape)

        return torch.view_as_complex(torch.stack((z[0], z[1]),dim=-1))


# Activations

def crelu(input: Tensor, inplace: bool = False) -> Tensor:
    '''
    Eq.(4)
    https://arxiv.org/pdf/1705.09792.pdf
    '''
    if input.is_complex():
        return torch.view_as_complex(torch.stack((F.relu(input.real, inplace=inplace), F.relu(input.imag, inplace=inplace)),dim=-1))
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

def modrelu(input: Tensor, bias: int, inplace: bool = False) -> Tensor:
    '''
    Martin Arjovsky, Amar Shah, and Yoshua Bengio. Unitary evolution recurrent neural networks. arXiv preprint arXiv:1511.06464, 2015.
    Notice that |z| (z.magnitude) is always positive, so if b > 0  then |z| + b > = 0 always.
    In order to have any non-linearity effect, b must be smaller than 0 (b<0).
    '''
    if input.is_complex():
        z_mag = torch.abs(input)
        return input * ((z_mag + bias) >= 0).float() * (1 + bias / z_mag)
    else:
        return F.relu(input, inplace=inplace)

def cmodrelu(input: Tensor, threshold: int, inplace: bool = False):
    r"""Compute the Complex modulus relu of the complex tensor in re-im pair.
    As proposed in : https://arxiv.org/pdf/1802.08026.pdf
    Source: https://github.com/ivannz/cplxmodule"""
    if input.is_complex():
        modulus = torch.clamp(torch.abs(input), min=1e-5)
        return input * F.relu(1. - threshold / modulus, inplace=inplace)
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