import torch
import torch.nn as nn
from torch.nn import functional as F

from neuralproc.utils.initialization import weights_init, init_param_
from neuralproc.utils.helpers import (make_depth_sep_conv, channels_to_2nd_dim,
                                      channel_to_last_dim)

__all__ = ["ConvBlock", "ResNormalizedConvBlock", "ResConvBlock", "CNN", "UnetCNN"]


class ConvBlock(nn.Module):
    """Simple convolutional block with a single layer.

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel.

    dilation : int or tuple, optional
        Spacing between kernel elements.

    padding : int or tuple, optional
        Padding added to both sides of the input. If `-1` uses padding that
        keeps the size the same. Currently only works if `kernel_size` is even
        and only takes into account the kenerl size and dilation, but not other
        arguments (e.g. stride).

    activation: callable, optional
        Activation object. E.g. `nn.ReLU`.

    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.

    is_normalized_conv : bool, optional
        Whether to use a normalized convolution [2], i.e. dividing by the
        convolution between the kernel and the confidence.

    kwargs :
        Additional arguments to `Conv`.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(self, in_chan, out_chan, Conv,
                 kernel_size=5,
                 dilation=1,
                 padding=-1,
                 activation=nn.ReLU(),
                 Normalization=nn.Identity,
                 is_normalized_conv=True,
                 **kwargs):
        super().__init__()
        self.activation = activation

        if padding == -1:
            padding = (kernel_size // 2) * dilation
            if kwargs.get("stride", 1) != 1:
                warnings.warn("`padding == -1` but `stride != 1`. The output might be of different dimension as the input depending on other hyperparameters.")

        if is_normalized_conv:
            Conv = make_depth_sep_conv(Conv)

        self.conv = Conv(in_chan, out_chan, kernel_size, padding=padding, **kwargs)
        self.norm = Normalization(out_chan)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        return self.conv(self.activation(self.norm(X)))


class ResConvBlock(nn.Module):
    """Convolutional block (2 layers) inspired by the pre-activation Resnet [1]
    and depthwise separable convolutions [2].

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.

    activation: callable, optional
        Activation object. E.g. `nn.RelU()`.

    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.

    is_bias : bool, optional
        Whether to use a bias.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.

    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(self, in_chan, out_chan, Conv,
                 kernel_size=5,
                 activation=nn.ReLU(),
                 Normalization=nn.Identity,
                 is_bias=True):
        super().__init__()
        self.activation = activation
        self.is_normalized_conv = is_normalized_conv

        if kernel_size % 2 == 0:
            raise ValueError("`kernel_size={}`, but should be odd.".format(kernel_size))

        padding = (kernel_size // 2)

        self.norm1 = Normalization(in_chan)
        self.conv1 = make_depth_sep_conv(Conv)(in_chan, in_chan, kernel_size,
                                               padding=padding,
                                               bias=is_bias)
        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(in_chan, in_chan, kernel_size,
                                    groups=in_chan, bias=is_bias)
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        out = self.conv1(self.activation(self.norm1(X)))
        out = self.conv2_depthwise(self.activation(self.norm2(X)))
        # adds residual before point wise => output can change number of channels
        out = out + X
        out = self.conv2_pointwise(out)
        return out


class ResNormalizedConvBlock(ResConvBlock):
    """Modification of `ResNormalizedConvBlock` to use normalized convolutions [1].

    Parameters
    ----------
    in_chan : int
        Number of input channels.

    out_chan : int
        Number of output channels.

    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.

    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.

    activation: nn.Module, optional
        Activation object. E.g. `nn.RelU()`.

    is_bias : bool, optional
        Whether to use a bias.

    References
    ----------
    [1] Knutsson, H., & Westin, C. F. (1993, June). Normalized and differential
        convolution. In Proceedings of IEEE Conference on Computer Vision and
        Pattern Recognition (pp. 515-523). IEEE.
    """

    def __init__(self, in_chan, out_chan, Conv,
                 kernel_size=5,
                 activation=nn.ReLU(),
                 is_bias=True):
        super().__init__(in_chan, out_chan, Conv,
                         kernel_size=kernel_size,
                         activation=activation,
                         is_bias=is_bias,
                         Normalization=nn.Identity)  # make sure no normalization

    def reset_parameters(self):
        weights_init(self)
        self.bias = nn.Parameter(torch.tensor([0.]))

        self.temperature = nn.Parameter(torch.tensor([0.]))
        self.temperature = init_param_(self.temperature)

    def forward(self, X):
        """
        Apply a normalized convolution. X should contain 2*in_chan channels.
        First halves for signal, last halve for corresponding confidence channels.
        """
        signal, conf_1 = X.split(2, dim=-1)
        # make sure confidence is in 0 1 (might not be due to the pointwise trsnf)
        conf_1 = conf_1.clamp(min=0, max=1)
        X = signal * conf_1
        numerator = self.conv1(self.activation(X))
        numerator = self.conv2_depthwise(self.activation(numerator))
        density = self.conv2_depthwise(self.conv1(conf_1))
        out = numerator / torch.clamp(density, min=1e-5)

        # adds residual before point wise => output can change number of channels

        # make sure that confidence cannot decrease and cannot be greater than 1
        conf_2 = conf_1 + torch.sigmoid(density *
                                        torch.softplus(self.temperature) +
                                        self.bias)
        conf_2 = conf_2.clamp(max=1)
        out = out + X

        out = self.conv2_pointwise(out)
        density_2 = self.conv2_pointwise(density_2)

        return torch.cat([out, density_2], dim=-1)


class CNN(nn.Module):
    """Simple multilayer CNN.

    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
        `[ConvBlock(16,32), ConvBlock(32, 64)]`.

    ConvBlock : nn.Module
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    n_blocks : int, optional
        Number of convolutional blocks.

    is_chan_last : bool, optional
        Whether the channels are on the last dimension of the input.

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(self, n_channels, ConvBlock,
                 n_blocks=3,
                 is_chan_last=False,
                 **kwargs):

        super().__init__()
        self.n_blocks = n_blocks
        self.is_chan_last = is_chan_last
        self.in_out_channels = self._get_in_out_channels(n_channels, n_blocks)
        self.conv_blocks = nn.ModuleList([ConvBlock(in_chan, out_chan, **kwargs)
                                          for in_chan, out_chan in self.in_out_channels])
        self.is_return_rep = False  # never return representation for vanilla conv

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(n_channels, int):
            channel_list = [n_channels] * n_blocks
        else:
            channel_list = list(n_channels)
            assert len(channel_list) == (n_blocks + 1), "{} != {}".format(len(channel_list), n_blocks + 1)

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):
        if self.is_chan_last:
            X = channels_to_2nd_dim(X)

        X, representation = self.apply_convs(X)

        if self.is_chan_last:
            X = channel_to_last_dim(X)

        if self.is_return_rep:
            return X, representation

        return X

    def apply_convs(self, X):
        for conv_block in self.conv_blocks:
            X = conv_block(X)
        return X, None


class UnetCNN(CNN):
    """Unet [?].

    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
        `[ConvBlock(16,32), ConvBlock(32, 64)]`.

    ConvBlock : nn.Module
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.

    Pool : nn.Module
        Pooling layer (unitialized). E.g. torch.nn.MaxPool1d.

    upsample_mode

    max_nchannels : int, optional
        Bounds the maximum number of channels instead of always doubling them at
        downsampling block.

    is_force_same_bottleneck : bool, optional
        Whether the channels are on the last dimension of the input.

    is_return_rep : bool, optional
        Whether to return a summary representation, that corresponds to the
        bottleneck + global mean pooling.

    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(self, n_channels, ConvBlock, Pool, upsample_mode,
                 max_nchannels=256,
                 is_force_same_bottleneck=False,
                 is_return_rep=False,
                 **kwargs):

        self.max_nchannels = max_nchannels
        super().__init__(n_channels, ConvBlock, **kwargs)
        self.pooling = Pool(self.pooling_size)
        self.upsample_mode = upsample_mode
        self.is_force_same_bottleneck = is_force_same_bottleneck
        self.is_return_rep = is_return_rep

    def apply_convs(self, X):
        n_down_blocks = n_blocks // 2
        residuals = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            X = self.convs[i](X)
            residuals[i] = X
            X = self.pooling(X)

        # Bottleneck
        X = self.convs[n_down_blocks](X)
        # Representation before forcing same bottleneck
        representation = X.view(*X.shape[:2], -1).mean(-1)

        if self.is_force_same_bottleneck and self.training:
            # forces the u-net to use the bottleneck by giving additional information
            # there. I.e. taking average between bottleenck of different samples
            # of the same functions. Because bottleneck should be a global representation
            # => should not depend on the sample you chose
            batch_size = X.size(0)
            batch_1 = X[:batch_size // 2, ...]
            batch_2 = X[batch_size // 2:, ...]
            X_mean = (batch_1 + batch_2) / 2
            X = torch.cat([X_mean, X_mean], dim=0)

        # Up
        for i in range(n_down_blocks + 1, n_blocks):
            X = F.interpolate(X,
                              mode=self.upsample_mode,
                              scale_factor=self.pooling_size,
                              align_corners=True)
            X = torch.cat((X, residuals[n_down_blocks - i]), dim=1)  # concat on channels
            X = self.convs[i](X)

        return X, representation

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels for a Unet."""
        # doubles at every down layer, as in vanilla U-net
        factor_chan = 2

        assert n_layers % 2 == 1, "n_blocks={} not odd".format(n_blocks)
        # e.g. if n_channels=16, n_blocks=5: [16, 32, 64]
        channel_list = [factor_chan**i * n_channels for i in range(n_blocks // 2 + 1)]
        # e.g.: [16, 32, 64, 64, 32, 16]
        channel_list = channel_list + channel_list[::-1]
        # bound max number of channels by self.max_nchannels
        channel_list = [min(c, self.max_nchannels) for c in channel_list]
        # e.g.: [(16, 32), (32,64), (64, 64), (64, 32), (32, 16)]
        in_out_channels = super()._get_in_out_channels(channel_list, n_blocks)
        # e.g.: [(16, 32), (32,64), (64, 64), (128, 32), (64, 16)] due to concat
        idcs = slice(len(in_out_channels) // 2 + 1, len(in_out_channels))
        in_out_channels[idcs] = [(in_chan * 2, out_chan)
                                 for in_chan, out_chan in in_out_channels[idcs]]
        return in_out_channels
