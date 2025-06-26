def _whiten2x2_group_norm(
    tensor: torch.Tensor,
    num_groups: int,
    normalized_shape,
    eps: float = 1e-5,
):
    r"""Group Normalisation Whitening
    -------------------------------

    Performs 2x2 whitening for group normalisation.

    This code has been adapted from the PyTorch implementation of LayerNorm and from github.
com/josiahwsmith10/complextorch/nn/functional.py from the following paper:
        **Smith, Josiah W. "Complex-valued neural networks for data-driven signal processing and signal
understanding." arXiv preprint arXiv:2309.07948 (2023)**
            - https://arxiv.org/abs/2309.07948
    """


    # assume tensor is 2 x B x F x ...
    assert tensor.dim() >= 3

    # Axes over which to compute mean and covariance
    axes = [-i - 1 for i in range(len(normalized_shape))]

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


def group_norm(input: torch.Tensor, num_groups: int, normalized_shape, weight=None, bias=None, eps=1e-5):
    r"""Group Normalisation
    -------------------------

    Extends the batch normalisation whitening definitions in the following paper:

        **J. A. Barrachina, C. Ren, G. Vieillard, C. Morisseau, and J.-P. Ovarlez. Theory and Implementation of
Complex-Valued Neural Networks.**
            - Section 6
            - https://arxiv.org/abs/2302.08286

    This code has been adapted from the PyTorch implementation of LayerNorm and from
github.com/josiahwsmith10/complextorch/nn/functional.py from the following paper:
        **Smith, Josiah W. "Complex-valued neural networks for data-driven signal processing and signal
understanding." arXiv preprint arXiv:2309.07948 (2023)**
            - https://arxiv.org/abs/2309.07948
    """


    # stack along the first axis
    input = torch.stack([input.real, input.imag], dim=0)

    # group whiten
    z = _whiten2x2_group_norm(
        input,
        num_groups,
        normalized_shape,
        eps=eps,
    )

    # apply affine transformation
    if weight is not None:
        shape = *[1] * (input.dim() - 1 - len(normalized_shape)), *normalized_shape
        weight = weight.view(2, 2, *shape)
        z = torch.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            dim=0,
        ) + bias.view(2, *shape)

    return torch.view_as_complex(torch.stack((z[0], z[1]), dim=-1))