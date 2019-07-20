import torch
import torch.nn as nn
from torch.nn import functional as F

from neuralproc.utils.initialization import weights_init
from neuralproc.utils.helpers import make_depth_sep_conv

__all__ = ["CNN", "UnetCNN", "SparseUnetCNN", "SparseCNN"]

# for confidence channel can try max pooling
class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, Conv,
                 kernel_size=5,
                 dilation=1,
                 padding=-1,
                 Activation=nn.ReLU,
                 is_depth_separable=False,
                 Normalization=nn.Identity,
                 confidence=None,
                 is_res=True,
                 bias=True,
                 **kwargs):
        super().__init__()
        self.activation_ = Activation(inplace=True)
        self.confidence = confidence
        self.is_bias = bias
        self.is_res = is_res

        if padding == -1:
            padding = (kernel_size // 2) * dilation

        kwargs.update(dict(dilation=dilation, padding=padding, bias=False))

        if self.confidence == "sparse":
            self.sparse_conv = Conv(in_chan, out_chan, kernel_size, **kwargs)
            self.sparse_conv.weight.requires_grad = False
            self.sparse_conv.weight.fill_(0.)
            self.pool = nn.MaxPool1d(kernel_size, padding=padding, stride=1)

        if is_depth_separable:
            Conv = make_depth_sep_conv(Conv)

        self.conv = Conv(in_chan, out_chan, kernel_size, **kwargs)

        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(out_chan).float(), requires_grad=True)

        self.norm = Normalization(out_chan)
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

        self.bias.data.fill_(0.)

        if self.confidence == "sparse":
            self.sparse_conv.weight.fill_(1.)

    def forward(self, X, confidence_channel=None):

        if self.confidence is not None:
            #confidence_channel = torch.sigmoid(X[..., -1:, :])
            X = confidence_channel * X

        out = self.conv(X)

        if self.confidence == "normalize":
            normalizer = self.conv(confidence_channel.expand_as(out))
            out = out / torch.clamp(normalizer, min=1e-5)
        elif self.confidence == "sparse":
            normalizer = self.sparse_conv(confidence_channel.expand_as(out))
            out = out / torch.clamp(normalizer, min=1e-5)
        elif self.confidence is not None:
            raise ValueError("Unkown confidence={}.".format(self.confidence))

        if self.is_bias:
            out = out + self.bias.view(1, -1, *[1 for _ in range(out.dim() - 2)])

        out = self.activation_(self.norm(out))

        if self.is_res:
            # maybe also needs to change based on confidence
            out = out + X

        if self.confidence == "sparse":
            confidence_channel = self.pool(confidence_channel)

        if confidence_channel is not None:
            return out, confidence_channel
        return out


class SparseCNN(nn.Module):
    def __init__(self, n_channels, Conv,
                 n_layers=3,
                 is_chan_last=False,
                 confidence=None,
                 _is_summary=False,
                 **kwargs):

        super().__init__()
        self._is_summary = _is_summary
        self.n_layers = n_layers
        self.is_chan_last = is_chan_last
        self.in_out_channels = self._get_in_out_channels(n_channels, n_layers)
        self.confidence = confidence

        self.convs = nn.ModuleList([ConvBlock(in_chan, out_chan, Conv,
                                              confidence=self.confidence, **kwargs)
                                    for in_chan, out_chan in self.in_out_channels])

        if self.confidence is not None:
            self.lin = nn.Linear(self.in_out_channels[-1][1] + 1, self.in_out_channels[-1][1])

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

    def forward(self, X, density=None):

        if self.is_chan_last:
            # put channels in second dim
            X = X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))

            if self.confidence is not None:
                density = density.permute(*([0, density.dim() - 1] +
                                            list(range(1, density.dim() - 1))))

        X = self.apply_convs(X, density)
        X, summary, density = X

        if not self._is_summary:
            summary = None

        if self.is_chan_last:
            # put back channels in last dim
            X = X.permute(*([0] + list(range(2, X.dim())) + [1]))

            if self.confidence is not None:
                density = density.permute(*([0] + list(range(2, density.dim())) + [1]))

        if self.confidence is not None:
            X = self.lin(torch.cat([X, density], dim=-1))

        return X, summary

    def apply_convs(self, X, density=None):
        for conv in self.convs:
            # normalization and residual
            if self.confidence is not None:
                X, density = conv(X, confidence_channel=density)
            else:
                X = conv(X, confidence_channel=density)

        return X, None, density


class SparseUnetCNN(SparseCNN):
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

    def apply_convs(self, X, density=None):

        n_blocks = self.n_layers // 2 if self.is_double_conv else self.n_layers
        n_down_blocks = n_blocks // 2
        residuals = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            if self.confidence is not None:
                X, density = self._apply_conv_block_i(X, i, density)
            else:
                X = self._apply_conv_block_i(X, i)

                residuals[i] = X
                X = self.pooling(X)

            if self.confidence is not None:
                density = self.pooling(density)

        # Bottleneck
        if self.confidence is not None:
            X, density = self._apply_conv_block_i(X, n_down_blocks, density)
        else:
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

            if self.confidence is not None:
                density = F.interpolate(density, mode=self.upsample_mode,
                                        scale_factor=self.pooling_size,
                                        align_corners=True)
                X, density = self._apply_conv_block_i(X, i, density)
            else:
                X = self._apply_conv_block_i(X, i)

        return X, summary, density

    def _apply_conv_block_i(self, X, i, density):
        """Apply the i^th convolution block."""
        if self.is_double_conv:
            i *= 2

        if self.confidence is not None:
            X, density = self.convs[i](X, confidence_channel=density)
        else:
            X = self.convs[i](X, confidence_channel=density)

        if self.is_double_conv:
            if self.confidence is not None:
                X, density = self.convs[i + 1](X, confidence_channel=density)
            else:
                X = self.convs[i + 1](X, confidence_channel=density)

        return X, None, density

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

    confidence : {"normalize", "sparse", None}, optional
        Whetehr to normlaize the convolution by a confidence measure, which
        should be in last channel of input. "normalize" is like a normalized
        convolution, i.e. dividing by th convolution between the kernel
        and the confidense. "sparse" is like sparse CNN which normalizes only
        by sum of confidence. `None` does not normalize.

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
                 confidence=None,
                 _is_summary=False,
                 **kwargs):

        super().__init__()
        self._is_summary = _is_summary
        self.n_layers = n_layers
        self.is_chan_last = is_chan_last
        self.activation_ = Activation(inplace=True)
        self.in_out_channels = self._get_in_out_channels(n_channels, n_layers)
        self.confidence = confidence

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
