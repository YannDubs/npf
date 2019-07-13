import torch
import torch.nn as nn
from torch.nn import functional as F

from neuralproc.utils.initialization import weights_init
from neuralproc.utils.helpers import make_depth_sep_conv

__all__ = ["CNN", "UnetCNN"]


class CNN(nn.Module):
    """Simple multilayer CNN class.

    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_layers - 1`, e.g. [16, 32, 64] means rhat you will have a
        `[Conv(16,32), Conv(32, 64)]`.

    Conv : _ConvNd
        Type of convolution to stack.

    n_layers : int, optional
        Number of convolutional layers.

    kernel_size : int, optional
        Size of the kernel to use at eacch layer.

    dilation : int, optional
        Spacing between kernel elements.

    padding : int, optional
        Zero-padding added to both sides of the input. If `-1` uses padding that
        keeps the size the same. Only possible if kernel_size is even. Currently
        only works if `kernel_size` is even and only takes into account the
        kenerl size and dilation,  but not other arguments (e.g. stride).

    activation: torch.nn.modules.activation, optional
        Unitialized activation class.

    is_chan_last : bool, optional
        Whether the channels are on the last dimension.

    is_depth_separable : bool, optional
        Whether to make the convolution depth separable. Which decreases the number
        of parameters and usually works well.

    Normalization : nn.Module, optional
        Normalization layer.

    kwargs :
        Additional arguments to `Conv`.
    """

    def __init__(self, n_channels, Conv,
                 n_layers=3,
                 kernel_size=5,
                 dilation=1,
                 padding=-1,
                 Activation=nn.ReLU,
                 is_chan_last=False,
                 is_depth_separable=False,
                 Normalization=nn.Identity,
                 _is_summary=False,
                 **kwargs):

        super().__init__()
        self._is_summary = _is_summary
        self.n_layers = n_layers
        self.is_chan_last = is_chan_last
        self.activation_ = Activation(inplace=True)
        self.in_out_channels = self._get_in_out_channels(n_channels, n_layers)

        if padding == -1:
            padding = (kernel_size // 2) * dilation

        if is_depth_separable:
            Conv = make_depth_sep_conv(Conv)

        self.convs = nn.ModuleList([Conv(in_chan, out_chan, kernel_size,
                                         dilation=dilation, padding=padding, **kwargs)
                                    for in_chan, out_chan in self.in_out_channels])

        self.norms = nn.ModuleList([Normalization(out_chan)
                                    for _, out_chan in self.in_out_channels])

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_in_out_channels(self, n_channels, n_layers):
        """Return a list of tuple of input and output channels."""

        if isinstance(n_channels, int):
            channel_list = [n_channels] * n_layers
        else:
            channel_list = list(n_channels)
            assert len(channel_list) == n_layers + 1

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):
        if self.is_chan_last:
            # put channels in second dim
            X = X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))

        X = self.apply_convs(X)
        X, summary = X

        if not self._is_summary:
            summary = None

        if self.is_chan_last:
            # put back channels in last dim
            X = X.permute(*([0] + list(range(2, X.dim())) + [1]))

        return X, summary

    def apply_convs(self, X):
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # normalization and residual
            X = self.activation_(norm(conv(X))) + X
        return X, None


class UnetCNN(CNN):
    def __init__(self, n_channels, Conv, Pool, upsample_mode,
                 is_double_conv=False,
                 bottleneck=None,  # latent or deterministic
                 n_layers=5,
                 **kwargs):

        self.is_double_conv = is_double_conv
        self.bottleneck = bottleneck
        super().__init__(n_channels, Conv,
                         padding=-1,
                         dilation=1,
                         n_layers=n_layers,
                         **kwargs)
        self.pooling_size = 4 if bottleneck is not None else 2
        self.pooling = Pool(self.pooling_size)
        self.upsample_mode = upsample_mode

    def apply_convs(self, X):

        n_blocks = self.n_layers // 2 if self.is_double_conv else self.n_layers
        n_down_blocks = n_blocks // 2
        residuals = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            X = self._apply_conv_block_i(X, i)
            residuals[i] = X
            X = self.pooling(X)

        # Bottleneck
        X = self._apply_conv_block_i(X, n_down_blocks)
        summary = X

        if self.bottleneck is not None:
            # if all the batches are from the same function then use the same
            # botlleneck for all to be sure that use the summary
            # X = X.mean(0, keepdim=True).expand(*X.shape)
            pass

        # Up
        for i in range(n_down_blocks + 1, n_blocks):
            X = F.interpolate(X, mode=self.upsample_mode, scale_factor=self.pooling_size,
                              align_corners=True)
            X = torch.cat((X, residuals[n_down_blocks - i]), dim=-2)  # conncat on channels
            X = self._apply_conv_block_i(X, i)

        return X, summary

    def _apply_conv_block_i(self, X, i):
        """Apply the i^th convolution block."""
        if self.is_double_conv:
            i *= 2

        X = self.activation_(self.norms[i](self.convs[i](X)))

        if self.is_double_conv:
            X = self.activation_(self.norms[i + 1](self.convs[i + 1](X)))

        return X

    def _get_in_out_channels(self, n_channels, n_layers):
        """Return a list of tuple of input and output channels for a Unet."""
        factor_chan = 1 if self.bottleneck is not None else 2

        if self.is_double_conv:
            assert n_layers % 2 == 0, "n_layers={} not even".format(n_layers)
            # e.g. if n_channels=16, n_layers=10: [16, 32, 64]
            channel_list = [factor_chan**i * n_channels for i in range(n_layers // 4 + 1)]
            # e.g.: [16, 16, 32, 32, 64, 64]
            channel_list = [i for i in channel_list for _ in (0, 1)]
            # e.g.: [16, 16, 32, 32, 64, 64, 64, 32, 32, 16, 16]
            channel_list = channel_list + channel_list[-2::-1]
            # e.g.: [16, 16, 32, 32, 64, 64, 64, 32, 32, 16, 16]
            # e.g.: [..., (32, 32), (32, 64), (64, 64), (64, 32), (32, 32), (32, 16) ...]
            in_out_channels = super()._get_in_out_channels(channel_list, n_layers)
            # e.g.: [..., (32, 32), (32, 64), (64, 64), (128, 32), (32, 32), (64, 16) ...]
            # due to concat
            idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels), 2)
            in_out_channels[idcs] = [(in_chan * 2, out_chan)
                                     for in_chan, out_chan in in_out_channels[idcs]]
        else:
            assert n_layers % 2 == 1, "n_layers={} not odd".format(n_layers)
            # e.g. if n_channels=16, n_layers=5: [16, 32, 64]
            channel_list = [factor_chan**i * n_channels for i in range(n_layers // 2 + 1)]
            # e.g.: [16, 32, 64, 64, 32, 16]
            channel_list = channel_list + channel_list[::-1]
            # e.g.: [(16, 32), (32,64), (64, 64), (64, 32), (32, 16)]
            in_out_channels = super()._get_in_out_channels(channel_list, n_layers)
            # e.g.: [(16, 32), (32,64), (64, 64), (128, 32), (64, 16)] due to concat
            idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
            in_out_channels[idcs] = [(in_chan * 2, out_chan)
                                     for in_chan, out_chan in in_out_channels[idcs]]

        return in_out_channels
