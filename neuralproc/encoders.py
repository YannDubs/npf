import torch
import torch.nn as nn

from neuralproc.utils.torchextend import MLP
from neuralproc.utils.initialization import weights_init
from neuralproc.utils.attention import get_attender

__all__ = ["SelfAttentionBlock", "SinusoidalEncodings", "get_uninitialized_mlp",
           "merge_flat_input"]


def get_uninitialized_mlp(**kwargs):
    return lambda *args, **kargs2: MLP(*args, **kwargs, **kargs2)


class SelfAttentionBlock(nn.Module):
    """Self Attention Layer.

    Parameters
    ----------
    x_dim : int
        Spatial input dimension.

    y_dim : int
        Value input dimension.

    out_dim : int
        Output dimension. If different than x_dim will do all the computation
        with a size of `x_dim` and add a linear layer at the end to reshape.

    n_attn_layers : int, optional
        Number of self attention layers.

    attention : {'multiplicative', "additive", "scaledot", "multihead", "manhattan",
                "euclidean", "cosine", "transformer"}, optional
        Type of attention to use. More details in `get_attender`.

    is_normalize : bool, optional
        Whether qttention weights should sum to 1 (using softmax). If not weights
        will be in [0,1] but not necessarily sum to 1.
    """

    def __init__(self, x_dim, y_dim, out_dim,
                 n_attn_layers=2,
                 attention="transformer",
                 is_normalize=True,
                 **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.out_dim = out_dim

        self.is_reshape_y = self.y_dim != self.x_dim
        if self.is_reshape_y:
            self.reshape_y = MLP(self.y_dim, self.x_dim)

        self.attn_layers = nn.ModuleList([get_attender(attention, self.x_dim,
                                                       is_normalize=is_normalize,
                                                       **kwargs)
                                          for _ in range(n_attn_layers)])

        self.is_reshape_out = self.x_dim != self.out_dim
        if self.is_reshape_out:
            self.reshape_out = nn.Linear(x_dim, self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X, Y):
        if self.is_reshape_y:
            Y = self.reshape_y(Y)

        out = Y + X

        for attn_layer in self.attn_layers:
            out = attn_layer(out, out, out)

        if self.is_reshape_out:
            out = self.reshape_out(out)

        return out


class SinusoidalEncodings(nn.Module):
    """
    Converts a batch of N-dimensional spatial input X with values between `[-1,1]`
    to a batch of flat in vector splitted in N subvectors that encode the position via
    sinusoidal encodings.

    Parameters
    ----------
    x_dim : int
        Number of spatial inputs.

    out_dim : int
        size of output encoding. Each x_dim will have an encoding of size
        `out_dim//x_dim`.
    """

    def __init__(self, x_dim, out_dim):
        super().__init__()
        self.x_dim = x_dim
        # dimension of encoding for eacg x dimension
        self.sub_dim = out_dim // self.x_dim
        # in attention is all you need used 10000 but 512 dim, try to keep the
        # same ratio regardless of dim
        self._C = 10000 * (self.sub_dim / 512)**2

        if out_dim % x_dim != 0:
            raise ValueError("out_dim={} has to be dividable by x_dim={}.".format(out_dim, x_dim))
        if self.sub_dim % 2 != 0:
            raise ValueError("sum_dim=out_dim/x_dim={} has to be dividable by 2.".format(self.sub_dim))

        self._precompute_denom()

    def _precompute_denom(self):
        two_i_d = torch.arange(0, self.sub_dim, 2, dtype=torch.float) / self.sub_dim
        denom = torch.pow(self._C, two_i_d)
        denom = torch.repeat_interleave(denom, 2).unsqueeze(0)
        self.denom = denom.expand(1, self.x_dim, self.sub_dim)

    def forward(self, x):
        shape = x.shape
        # flatten besides last dim
        x = x.view(-1, shape[-1])
        # will only be passed once to GPU because precomputed
        self.denom = self.denom.to(x.device)
        # put x in a range which is similar to positions in NLP [1,201]
        #x = (x.unsqueeze(-1) + 1)*100 + 1
        x = x.unsqueeze(-1)
        out = x / self.denom
        out[:, :, 0::2] = torch.sin(out[:, :, 0::2])
        out[:, :, 1::2] = torch.cos(out[:, :, 1::2])
        # concatenate all different sinusoidal encodings for each x_dim
        # and unflatten
        out = out.view(*shape[:-1], self.sub_dim * self.x_dim)
        return out


# META ENCODERS
class MergeFlatAndNotFlatInputs(nn.Module):
    """
    Extend a module which takes takes a non flat input (e.g. images) such that it can
    also take a flat input. The output of the given module will be concatenated with
    the flat input and passed to a MLP.

    Parameters
    ----------
    NonFlatModule: nn.Module
        Module which takes in non flat inputs (e.g. images).

    x_shape: array-like
        Shape of a single non flat example.

    flat_dim: int
        Dimensionality of the flat inputs.

    n_out: int
        Size of ouput.

    hidden_size: int, optional
        Hidden size of the MLP which puts together the output of NonFlatModule and the
        flat input. If -1 uses n_out.

    is_sum_merge : bool, optional
        Whether to transform `flat_input` by an MLP first, then sum to `X`
        (instead of concatenating): useful if the difference in dimension
        between both inputs is very large => don't want one layer to depend only on
        a few dimension of a large input.

    kwargs:
        Additional arguments to NonFlatModule.
    """

    def __init__(self, NonFlatModule, x_shape, flat_dim, n_out,
                 hidden_size=-1,
                 n_hidden_layers=1,
                 **kwargs):
        super().__init__()
        hidden_size = hidden_size if hidden_size != -1 else n_out
        self.non_flat_module = NonFlatModule(x_shape, hidden_size, **kwargs)

        self.is_sum_merge = is_sum_merge
        if self.is_sum_merge:
            # transform then sum
            self.resizer = MLP(flat_dim, hidden_size)
            self.mixer = MLP(hidden_size, n_out,
                             hidden_size=hidden_size,
                             n_hidden_layers=n_hidden_layers)
        else:
            # concat
            self.mixer = MLP(hidden_size + flat_dim, n_out,
                             hidden_size=hidden_size,
                             n_hidden_layers=n_hidden_layers)
        self.reset_parameters()

    def forward(self, x, flat_input):
        non_flat_out = self.non_flat_module(x)
        if self.is_sum_merge:
            flat_input = self.resizer(flat_input)
            # use activation becaus eif not 2 linear layers in a row => useless computation
            out = self.resizer.activation(non_flat_out + flat_input)
        else:
            out = torch.cat((non_flat_out, flat_input), dim=-1)

        return self.mixer(out)

    def reset_parameters(self):
        weights_init(self)


class MergeFlatInputs(nn.Module):
    """
    Extend a module to take 2 flat inputs. It simply returns
    the concatenated flat inputs to the module `module({x1; x2})`.

    Parameters
    ----------
    FlatModule: nn.Module
        Module which takes a non flat inputs.

    x1_dim: int
        Dimensionality of the first flat inputs.

    x2_dim: int
        Dimensionality of the second flat inputs.

    n_out: int
        Size of ouput.

    is_sum_merge : bool, optional
        Whether to transform `flat_input` by an MLP first (if need to resize),
        then sum to `X` (instead of concatenating): useful if the difference in
        dimension between both inputs is very large => don't want one layer to
        depend only on a few dimension of a large input.

    kwargs:
        Additional arguments to FlatModule.
    """

    def __init__(self, FlatModule, x1_dim, x2_dim, n_out,
                 is_sum_merge=False,
                 **kwargs):
        super().__init__()
        self.is_sum_merge = is_sum_merge

        if self.is_sum_merge:
            # transform then sum
            self.resizer = MLP(x2_dim, x1_dim)
            self.flat_module = FlatModule(x1_dim, n_out, **kwargs)
        else:
            # concat
            self.flat_module = FlatModule(x1_dim + x2_dim, n_out, **kwargs)
        self.reset_parameters()

    def forward(self, x1, x2):
        if self.is_sum_merge:
            x2 = self.resizer(x2)
            # use activation becaus eif not 2 linear layers in a row => useless computation
            out = self.resizer.activation(x1 + x2)
        else:
            out = torch.cat((x1, x2), dim=-1)

        return self.flat_module(out)

    def reset_parameters(self):
        weights_init(self)


def merge_flat_input(module, is_sum_merge=False, **kwargs):
    """
    Extend a module to accept an additional flat input. I.e. the output should
    be called by `merge_flat_input(module)(x_shape, flat_dim, n_out, **kwargs)`.

    Notes
    -----
    - if x_shape is an integer (i.e. x is already flat), it simply returns
    the concatenated flat inputs to the module `module({x; flat_input})`.
    - if x_shape is not an integer (i.e. non flat input), it concatenates the
    output of `module(x)` with the flat input and passes it through a MLP.
    - if `is_sum_merge` then transform `flat_input` by an MLP first, then sum
    to `X` (instead of concatenating): useful if the difference in dimension
    between both inputs is very large => don't want one layer to depend only on
    a few dimension of a large input.
    """
    def merged_flat_input(x_shape, flat_dim, n_out, **kwargs2):
        if isinstance(x_shape, int):
            return MergeFlatInputs(module, x_shape, flat_dim, n_out,
                                   is_sum_merge=is_sum_merge, **kwargs2, **kwargs)
        else:
            return MergeFlatAndNotFlatInputs(module, x_shape, flat_dim, n_out,
                                             is_sum_merge=is_sum_merge, **kwargs2, **kwargs)
    return merged_flat_input
