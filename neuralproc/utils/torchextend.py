import warnings


import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.independent import Independent
from torch.distributions import Normal

from .initialization import linear_init


class MLP(nn.Module):
    """General MLP class.

    Parameters
    ----------
    input_size: int

    output_size: int

    hidden_size: int, optional
        Number of hidden neurones.

    n_hidden_layers: int, optional
        Number of hidden layers.

    activation: torch.nn.modules.activation, optional
        Unitialized activation class.

    bias: bool, optional
        Whether to use biaises in the hidden layers.

    dropout: float, optional
        Dropout rate.
    """

    def __init__(self, input_size, output_size,
                 hidden_size=32,
                 n_hidden_layers=1,
                 activation=nn.ReLU,
                 bias=True,
                 dropout=0):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        if self.hidden_size < min(self.output_size, self.input_size):
            self.hidden_size = min(self.output_size, self.input_size)
            txt = "hidden_size={} smaller than output={} and input={}. Setting it to {}."
            warnings.warn(txt.format(hidden_size, output_size, input_size, self.hidden_size))

        self.dropout = (nn.Dropout(p=dropout) if dropout > 0 else identity)
        self.activation = activation()  # cannot be a function from Functional but class

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
                                      for _ in range(self.n_hidden_layers - 1)])
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        out = self.to_hidden(x)
        out = self.activation(out)
        out = self.dropout(out)

        for linear in self.linears:
            out = linear(out)
            out = self.activation(out)
            out = self.dropout(out)

        out = self.out(out)
        return out

    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation)
        for lin in self.linears:
            linear_init(lin, activation=self.activation)
        linear_init(self.out)


def identity(x):
    """simple identity function"""
    return x


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
