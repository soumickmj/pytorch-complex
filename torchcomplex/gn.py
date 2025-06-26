
def _whiten2x2_group_norm(tensor, num_groups, eps=1e-5):
    """
    Performs 2x2 whitening for group normalisation on complex-valued tensors.
    """
    # Check for channel dimension and divisibility by num_groups
    assert tensor.dim() >= 3
    C = tensor.size(1)
    assert C % num_groups == 0, "num_channels must be divisible by num_groups"

    group_size = C // num_groups
    original_shape = tensor.shape
    tensor = tensor.view(2, num_groups, group_size, *tensor.shape[2:])

    # Compute mean and variance within groups
    mean = tensor.mean(dim=[2, 3], keepdim=True)
    tensor -= mean
    var = (tensor * tensor).mean(dim=[2, 3], keepdim=True) + eps

    v_rr, v_ii = var[0][0], var[1][1]
    v_ir = (tensor[0] * tensor[1]).mean(dim=[2, 3], keepdim=True)[0]

    # Compute inverse square root of the covariance matrix
    p, q, _, s = inv_sqrtm2x2(v_rr, v_ir, v_ir, v_ii)

    # Whiten the tensor within each group
    whitened = torch.stack([
        tensor[0] * p + tensor[1] * q,
        tensor[0] * q + tensor[1] * s
    ], dim=0)

    return whitened.view_as(original_shape)

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """
    Applies complex-valued group normalization by dividing the channels into specified groups.
    """
    input_stacked = torch.stack([input.real, input.imag], dim=0)
    z = _whiten2x2_group_norm(input_stacked, num_groups, eps)

    if weight is not None and bias is not None:
        # Reshape weight and bias
        weight_shape = (2, num_groups, *([1] * (input.dim() - 2)))
        bias_shape = (2, num_groups, *([1] * (input.dim() - 2)))
        weight = weight.view(*weight_shape)
        bias = bias.view(*bias_shape)

        z = torch.stack([
            z[0] * weight[0] + z[1] * weight[1],
            z[0] * weight[1] + z[1] * weight[0]
        ], dim=0) + bias

    # Convert back to a complex tensor and return
    return torch.view_as_complex(torch.stack((z[0], z[1]), dim=-1))

