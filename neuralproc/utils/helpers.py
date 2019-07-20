from functools import reduce
import operator

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.independent import Independent
from torch.distributions import Normal

from .initialization import weights_init


def mask_and_apply(x, mask, f):
    """Applies a callable on a masked version of a input."""
    tranformed_selected = f(x.masked_select(mask))
    return x.masked_scatter(mask, tranformed_selected)

# SHOULD USE partial from functools !
def change_param(callable, **kwargs):
    def changed_callable(*args, **kwargs2):
        return callable(*args, **kwargs, **kwargs2)
    return changed_callable


def indep_shuffle_(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.

    Credits : https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])


def ratio_to_int(percentage, max_val):
    """Converts a ratio to an integer if it is smaller than 1."""
    if 1 <= percentage <= max_val:
        out = percentage
    elif 0 <= percentage < 1:
        out = percentage * max_val
    else:
        raise ValueError("percentage={} outside of [0,{}].".format(percentage, max_val))

    return int(out)


def prod(iterable):
    """Compute the product of all elements in an iterable."""
    return reduce(operator.mul, iterable, 1)


def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min


def min_max_scale(tensor, min_val=0, max_val=1, dim=0):
    """Rescale value to be in a given range across dim."""
    tensor = tensor.float()
    std_tensor = (tensor - tensor.min(dim=dim, keepdim=True)[0]
                  ) / (tensor.max(dim=dim, keepdim=True)[0] - tensor.min(dim=dim, keepdim=True)[0])
    scaled_tensor = std_tensor * (max_val - min_val) + min_val
    return scaled_tensor


def MultivariateNormalDiag(loc, scale_diag):
    if loc.dim() < 1:
        raise ValueError("loc must be at least one-dimensional.")
    return Independent(Normal(loc, scale_diag), 1)


def make_depth_sep_conv(Conv):
    """Make a convolution module depth separable."""
    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.

        Parameters
        ----------
        in_channels : int
            Number of input channels.

        out_channels : int
            Number of output channels.

        kernel_size : int

        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(self, in_channels, out_channels, kernel_size,
                     confidence=False, bias=True, **kwargs):
            super().__init__()
            self.depthwise = Conv(in_channels, in_channels, kernel_size,
                                  groups=in_channels, bias=bias, **kwargs)
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv


class BackwardPDB(torch.autograd.Function):
    """Run PDB in the backward pass."""
    @staticmethod
    def forward(ctx, input, name="debugger"):
        ctx.name = name
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        if not torch.isfinite(grad_output).all() or not torch.isfinite(input).all():
            import pdb
            pdb.set_trace()
        return grad_output, None  # 2 args so return None for `name`


backward_pdb = BackwardPDB.apply
