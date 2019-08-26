import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.independent import Independent
from torch.distributions import Normal

from neuralproc.predefined import MLP, CNN, UnetCNN, ResConvBlock
from neuralproc.utils.initialization import weights_init, init_param_
from neuralproc.utils.helpers import (MultivariateNormalDiag, ProbabilityConverter,
                                      make_abs_conv, channels_to_2nd_dim,
                                      channel_to_last_dim)
from neuralproc.utils.datasplit import CntxtTrgtGetter
from neuralproc.utils.attention import get_attender
from neuralproc.utils.setcnn import SetConv, GaussianRBF

from .encoders import merge_flat_input, discard_ith_arg, RelativeSinusoidalEncodings


__all__ = ["NeuralProcess", "AttentiveNeuralProcess", "ConvolutionalProcess",
           "RegularGridsConvolutionalProcess"]


class NeuralProcess(nn.Module):
    """
    Implements (Conditional [2]) Neural Process [1] using tricks from [3] for
    functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    r_dim : int, optional
        Dimension of representation.

    x_transf_dim : int, optonal
        Dimension of the encoded X. If `-1` uses `r_dim`. if `None` uses `x_dim`.

    XEncoder : nn.Module, optional
        Spatial encoder module which maps {x_i} -> x_transf_i. It should be
        constructable via `XEncoder(x_dim, x_transf_dim)`. `None` uses
        parameter dependent default. Example:
            - `MLP` : will learn positional embeddings with MLP
            - `SinusoidalEncodings` : use sinusoidal positional encodings.

    XYEncoder : nn.Module, optional
        Encoder module which maps {x_transf_i, y_i} -> {r_i}. It should be constructable
        via `XYEncoder(x_transf_dim, y_dim, n_out)`. If you have an encoder that maps
        [x;y] -> r you can convert it via `merge_flat_input(Encoder)`. `None` uses
        parameter dependent default. Example:
            - `merge_flat_input(MLP, is_sum_merge=False)` : learn representation
            with MLP. `merge_flat_input` concatenates (or sums) X and Y inputs.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : self attention
            mechanisms as [4]. For more parameters (attention type, number of
            layers ...) refer to its docstrings.
            - `discard_ith_arg(MLP, 0)` if want the encoding to only depend on Y.

    Decoder : nn.Module, optional
        Decoder module which maps {x_t, r} -> {y_hat_t}. It should be constructable
        via `decoder(x, r_dim, n_out)`. If you have an decoder that maps
        [r;x] -> y you can convert it via `merge_flat_input(Decoder)`. `None` uses
        parameter dependent default. Example:
            - `merge_flat_input(MLP)` : predict with MLP.
            - `merge_flat_input(SelfAttention, is_sum_merge=True)` : predict
            with self attention mechanisms (using `X_transf + Y` as input) to have
            coherant predictions (not use in attentive neural process [4] but in
            image transformer [5]).
            - `discard_ith_arg(MLP, 0)` if want the decoding to only depend on r.

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suff_stat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`. Only used if `encoded_path in ["latent",
        "both"]`.

    encoded_path : {"deterministic", "latent", "both"}
        Which path(s) to use:
        - `"deterministic"` uses a Conditional Neural Process [2] (no latents),
        where the decoder gets a deterministic representation as input
        (function of the context).
        - `"latent"` uses the original Neural Process [1], where the decoder gets
        a sample latent representation as input (function of the target during
        training and context during test).
        - `"both"` concatenates both representations as described in [4].

    PredictiveDistribution : torch.distributions.Distribution, optional
        Predictive distribution. The predicted outputs will be independent and thus
        wrapped around `Independent` (e.g. diagonal covariance for a Gaussian).
        The input to the constructor are currently a value in ]-inf, inf[ and one
        in [min_std, inf[ (typically `loc` and `scale`).

    is_use_x : bool, optional
        Whether to encode and use X in the representation (r_i) and when decoding.
        If `False`, then guarantees translation equivariance (if add some
        representation of the positional differences) or invariance.

    pred_loc_transformer : callable, optional
        Transformation to apply to the predicted location (e.g. mean for Gaussian)
        of Y_trgt.

    pred_scale_transformer : callable, optional
        Transformation to apply to the predicted scale (e.g. std for Gaussian) of
        Y_trgt. The default follows [3] by using a minimum of 0.1.

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    [2] Garnelo, Marta, et al. "Conditional neural processes." arXiv preprint
        arXiv:1807.01613 (2018).
    [3] Le, Tuan Anh, et al. "Empirical Evaluation of Neural Process Objectives."
        NeurIPS workshop on Bayesian Deep Learning. 2018.
    [4] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    [5] Parmar, Niki, et al. "Image transformer." arXiv preprint arXiv:1802.05751
        (2018).
    """

    def __init__(self, x_dim, y_dim,
                 r_dim=128,
                 x_transf_dim=-1,
                 XEncoder=None,
                 XYEncoder=None,
                 Decoder=None,
                 LatentEncoder=MLP,
                 encoded_path="deterministic",
                 PredictiveDistribution=Normal,
                 is_use_x=True,
                 pred_loc_transformer=nn.Identity(),
                 pred_scale_transformer=lambda scale_trgt: 0.1 + 0.9 * F.softplus(scale_trgt)
                 ):
        super().__init__()

        Decoder, XYEncoder, x_transf_dim, XEncoder = self._get_defaults(Decoder,
                                                                        XYEncoder,
                                                                        x_transf_dim,
                                                                        XEncoder,
                                                                        is_use_x,
                                                                        r_dim)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path.lower()
        self.PredictiveDistribution = PredictiveDistribution
        self.pred_loc_transformer = pred_loc_transformer
        self.pred_scale_transformer = pred_scale_transformer

        if x_transf_dim is None:
            self.x_transf_dim = self.x_dim
        elif x_transf_dim == -1:
            self.x_transf_dim = self.r_dim
        else:
            self.x_transf_dim = x_transf_dim

        self.x_encoder = XEncoder(self.x_dim, self.x_transf_dim)
        self.xy_encoder = XYEncoder(self.x_transf_dim, self.y_dim, self.r_dim)
        # *2 because mean and var
        self.decoder = Decoder(self.x_transf_dim, self.r_dim, self.y_dim * 2)

        if self.encoded_path in ["latent", "both"]:
            self.lat_encoder = LatentEncoder(self.r_dim, self.r_dim * 2)
            if self.encoded_path == "both":
                self.merge_rz = nn.Linear(self.r_dim * 2, self.r_dim)
        elif self.encoded_path == "deterministic":
            self.lat_encoder = None
        else:
            raise ValueError("Unkown encoded_path={}.".format(encoded_path))

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_defaults(self, Decoder, XYEncoder, x_transf_dim, XEncoder, is_use_x, r_dim):
        dflt_sub_decoder = partial(MLP, n_hidden_layers=4, is_force_hid_smaller=True)
        dflt_sub_xyencoder = partial(MLP, n_hidden_layers=2, is_force_hid_smaller=True)

        # don't use `x` to be translation equivariant
        if not is_use_x:
            if Decoder is None:
                Decoder = discard_ith_arg(dflt_sub_decoder, i=0)  # depend only on r not x

            if XYEncoder is None:
                XYEncoder = discard_ith_arg(dflt_sub_xyencoder, i=0)  # depend only on y not x

            if XEncoder is None:
                # don't encode X if not using it
                x_transf_dim = None
                XEncoder = nn.Identity
        else:
            if Decoder is None:
                Decoder = merge_flat_input(dflt_sub_decoder, is_sum_merge=True)
            if XYEncoder is None:
                XYEncoder = merge_flat_input(dflt_sub_xyencoder, is_sum_merge=True)
            if XEncoder is None:
                XEncoder = MLP

        return Decoder, XYEncoder, x_transf_dim, XEncoder

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        """
        Given a set of context pairs {x_i, y_i} and target points {x_t}, return
        a set of posterior distribution over target points {y_trgt}.

        Parameters
        ----------
        X_cntxt: torch.Tensor, size=[batch_size, n_cntxt, x_dim]
            Set of all context features {x_i}.

        Y_cntxt: torch.Tensor, size=[batch_size, n_cntxt, y_dim]
            Set of all context values {y_i}.

        X_trgt: torch.Tensor, size=[batch_size, n_trgt, x_dim]
            Set of all target features {x_t}.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training and if
            using latent path.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution
            Target distribution.

        q_z_trgt: torch.distributions.Distribution
            Latent distribution for the targets. `None` if `LatentEncoder=None`
            or not training.

        q_z_cntxt: torch.distributions.Distribution
            Latent distribution for the context points. `None` if
            `LatentEncoder=None` or not training.
        """

        # input assumed to be in [-1,1] during training
        if self.training:
            if X_cntxt.max() > 1 or X_cntxt.min() < -1 or X_trgt.max() > 1 and X_trgt.min() < -1:
                raise ValueError("Position inputs during training should be in [-1,1]. {} < X_cntxt < {} ; {} < X_trgt < {}.".format(X_cntxt.min(), X_cntxt.max(), X_trgt.min(), X_trgt.max()))

        R_det, z_sample, q_z_cntxt, q_z_trgt = None, None, None, None

        if self.encoded_path in ["latent", "both"]:
            z_sample, q_z_cntxt = self.latent_path(X_cntxt, Y_cntxt)

            if self.training:
                # during training when we know Y_trgt, we compute the latent using
                # the targets as context (which also contain the context). If we
                # used it for the deterministic path, then the model would cheat
                # by learning a point representation for each function => bad representation
                z_sample, q_z_trgt = self.latent_path(X_trgt, Y_trgt)

        if self.encoded_path in ["deterministic", "both"]:
            R_det = self.deterministic_path(X_cntxt, Y_cntxt, X_trgt)

        dec_inp = self.make_dec_inp(R_det, z_sample, X_trgt)
        p_y_trgt = self.decode(dec_inp, X_trgt)

        return p_y_trgt, q_z_trgt, q_z_cntxt

    def latent_path(self, X, Y):
        """Latent encoding path."""
        # size = [batch_size, n_cntxt, x_transf_dim]
        X_transf = self.x_encoder(X)
        # size = [batch_size, n_cntxt, x_transf_dim]
        R_cntxt = self.xy_encoder(X_transf, Y)

        # size = [batch_size, r_dim]
        r = torch.mean(R_cntxt, dim=1)

        z_suff_stat = self.lat_encoder(r)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives".
        mean_z, std_z = z_suff_stat.view(z_suff_stat.shape[0], -1, 2).unbind(-1)
        std_z = 0.1 + 0.9 * torch.sigmoid(std_z)
        # use a Gaussian prior on latent
        q_z = MultivariateNormalDiag(mean_z, std_z)
        z_sample = q_z.rsample()

        return z_sample, q_z

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path. `X_trgt` can be used in child classes
        to give a target specific representation (e.g. attentive neural processes).
        """

        # size = [batch_size, n_cntxt, x_transf_dim]
        X_transf = self.x_encoder(X_cntxt)
        # size = [batch_size, n_cntxt, x_transf_dim]
        R_cntxt = self.xy_encoder(X_transf, Y_cntxt)

        # size = [batch_size, r_dim]
        r = torch.mean(R_cntxt, dim=1)

        batch_size, n_trgt, _ = X_trgt.shape
        R = r.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
        return R

    def make_dec_inp(self, R, z_sample, X_trgt):
        """Make the context input for the decoder."""
        batch_size, n_trgt, _ = X_trgt.shape

        if self.encoded_path == "both":
            Z = z_sample.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
            dec_inp = torch.relu(self.merge_rz(torch.cat((R, Z), dim=-1)))
        elif self.encoded_path == "latent":
            Z = z_sample.unsqueeze(1).expand(batch_size, n_trgt, self.r_dim)
            dec_inp = Z
        elif self.encoded_path == "deterministic":
            dec_inp = R

        return dec_inp

    def decode(self, dec_inp, X_trgt):
        """
        Compute predicted distribution conditioned on representation and
        target positions.

        Parameters
        ----------
        dec_inp : torch.Tensor, size=[batch_size, n_trgt, inp_dim]
            Input to the decoder. `inp_dim` is `r_dim * 2 + x_dim` if
            `encoded_path == "both"` else `r_dim + x_dim`.

        X_trgt: torch.Tensor, size=[batch_size, n_trgt, x_dim]
            Set of all target features {x_t}.
        """
        # size = [batch_size, n_trgt, x_transf_dim]
        X_transf = self.x_encoder(X_trgt)

        # size = [batch_size, n_trgt, y_dim*2]
        suff_stat_Y_trgt = self.decoder(X_transf, dec_inp)

        loc_trgt, scale_trgt = suff_stat_Y_trgt.split(self.y_dim, dim=-1)

        loc_trgt = self.pred_loc_transformer(loc_trgt)
        scale_trgt = self.pred_scale_transformer(scale_trgt)

        p_y = Independent(self.PredictiveDistribution(loc_trgt, scale_trgt), 1)

        return p_y

    def set_extrapolation(self, min_max):
        """Set the neural process for extrapolation. Useful for child classes."""
        pass

    def preprocess_inputs(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt):
        """Preprocesses the inputs. Useful for child classes."""
        return X_cntxt, Y_cntxt, X_trgt, Y_trgt


class AttentiveNeuralProcess(NeuralProcess):
    """
    Wrapper around `NeuralProcess` that implements an attentive neural process [1].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    attention : callable or str, optional
        Type of attention to use. More details in `get_attender`.

    is_translation_equiv : bool, optional
        Whether to make the attentive neural process translation equivariant.

    attention_kwargs : dict, optional
        Additional arguments to `get_attender`.

    kwargs :
        Additional arguments to `NeuralProcess`.

    References
    ----------
    [1] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    def __init__(self, x_dim, y_dim,
                 attention="scaledot",
                 is_translation_equiv=False,
                 encoded_path="both",
                 attention_kwargs={},
                 **kwargs):

        self.is_translation_equiv = is_translation_equiv
        if self.is_translation_equiv:
            attention_kwargs["is_relative_pos"] = True
            kwargs["is_use_x"] = False
            # transform even though not using => the input will be of the correct size
            kwargs["XEncoder"] = nn.Linear

        super().__init__(x_dim, y_dim, encoded_path=encoded_path, **kwargs)

        self.attender = get_attender(attention, self.x_transf_dim, self.r_dim,
                                     self.r_dim, **attention_kwargs)

        if self.is_translation_equiv:
            self.rel_pos_encoder = RelativeSinusoidalEncodings(x_dim, self.r_dim)

        self.reset_parameters()

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):

        # size = [batch_size, n_cntxt, x_transf_dim]
        keys = self.x_encoder(X_cntxt)

        # size = [batch_size, n_trgt, r_dim]
        queries = self.x_encoder(X_trgt)

        if self.is_translation_equiv:
            # dirty trick such that keys and queries do not have positioning
            # information but are the correct size.
            keys, queries = keys * 0., queries * 0.

            # size = [batch_size, n_queries, n_keys, kq_size]
            attender_kwargs = dict(rel_pos_enc=self.rel_pos_encoder(X_cntxt, X_trgt))
        else:
            attender_kwargs = {}

        # size = [batch_size, n_cntxt, r_dim]
        values = self.xy_encoder(keys, Y_cntxt)

        # size = [batch_size, n_trgt, r_dim]
        R_attn = self.attender(keys, queries, values, **attender_kwargs)

        return R_attn


class ConvolutionalProcess(NeuralProcess):
    """
    Convolutional Process [Jonathan]. I.e. with temporary queries to represent the
    functional space.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    n_pseudo : int, optional
        Number of pseudo-inputs (temporary queries) to use. The pseudo-inputs
        will be regularaly sampled.

    inp_to_pseudoinp : callable or str, optional
        Callable or name of attention to use for {inp} -> {pseudo_inp}.
        More details in `get_attender`. Example:
            - `SetConv` : uses a set convolution as in the paper.
            - `"transformer"` : uses a cross attention layer as in the transformer.

    PseudoTransformer : callable, optional
        Object used to transform the pseudo inputs {pseudo_inp} -> {pseudo_inp}. Note
        that the temporary queries will be uniformly sampled and you can thus use
        a convolution instead. Example:
            - `partial(CNN, is_chan_last=True)` : uses a multilayer CNN. To
            be compatible with self attention the channel layer should be last.
            - `SelfAttention` : uses a self attention layer.

    pseudoinp_to_out : callable or str, optional
        Callable or name of attention to use for {pseudo_inp} -> {r_trgt}.
        More details in `get_attender`. Example:
            - `SetConv` : uses a set convolution as in the paper.
            - `"transformer"` : uses a cross attention layer as in the transformer.

    is_use_x : bool, optional
        Whether to encode and use X in the representation (r_i) and when decoding.
        If `False`, then guarantees translation equivariance (if add some
        representation of the positional differences) or invariance.

    is_encode_xy : bool, optional
        Whether to encode x and y.

    kwargs :
        Additional arguments to `NeuralProcess`.

    References
    ----------
    [Jonathan]
    """

    def __init__(self, x_dim, y_dim,
                 n_pseudo=256,
                 keys_to_pseudo=SetConv,
                 PseudoSelfAttn=partial(CNN, ResConvBlock,
                                        Conv=nn.Conv1d,
                                        n_blocks=3,
                                        Normalization=nn.Identity,
                                        is_chan_last=True,
                                        kernel_size=11),
                 pseudo_to_queries=SetConv,
                 **kwargs):

        super().__init__(x_dim, y_dim,
                         encoded_path="deterministic",
                         is_use_x=False,
                         XYEncoder=nn.Identity,
                         **kwargs)

        self.n_pseudo = n_pseudo
        self.pseudo_keys = torch.linspace(-1, 1, self.n_pseudo)
        self.keys_to_pseudo = get_attender(keys_to_pseudo, self.x_dim,
                                           self.y_dim, self.r_dim)
        self.pseudo_self_attn = PseudoSelfAttn(self.r_dim)
        self.pseudo_to_queries = get_attender(pseudo_to_queries, self.x_dim,
                                              self.r_dim, self.r_dim)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path.
        """
        # effectively puts on cuda only once
        self.pseudo_keys = self.pseudo_keys.to(X_cntxt.device)
        pseudo_keys = self.pseudo_keys.view(1, -1, 1)

        keys, values, queries = X_cntxt, Y_cntxt, X_trgt

        # size = [batch_size, n_pseudo, x_dim]
        pseudo_keys = pseudo_keys.expand(X_cntxt.size(0),
                                         pseudo_keys.size(1),
                                         self.x_dim)

        # size = [batch_size, n_pseudo, r_dim]
        pseudo_values = self.keys_to_pseudo(keys, pseudo_keys, values)
        pseudo_values = torch.relu(pseudo_values)
        pseudo_values = self.pseudo_self_attn(pseudo_values)
        pseudo_values = torch.relu(pseudo_values)

        # size = [batch_size, n_trgt, r_dim]
        R_attn = self.pseudo_to_queries(pseudo_keys, queries, pseudo_values)

        return torch.relu(R_attn)

    def set_extrapolation(self, min_max):
        """
        Scale the pseudo keys to be in a given range while keeping
        the same density than during training (used for extrapolation.).
        """
        self.pseudo_keys = torch.linspace(-1, 1, self.n_pseudo)  # reset
        current_min = -1
        current_max = 1

        delta = self.pseudo_keys[1] - self.pseudo_keys[0]
        n_queries_per_increment = self.n_pseudo / (current_max - current_min)
        n_add_left = math.ceil((current_min - min_max[0]) * n_queries_per_increment)
        n_add_right = math.ceil((min_max[1] - current_max) * n_queries_per_increment)

        tmp_queries_l = []
        if n_add_left > 0:
            tmp_queries_l.append(torch.arange(min_max[0], current_min, delta))

        tmp_queries_l.append(self.tmp_queries)

        if n_add_right > 0:
            # add delta to not have twice the previous max boundary
            tmp_queries_l.append(torch.arange(current_max, min_max[1], delta) + delta)

        self.tmp_queries = torch.cat(tmp_queries_l)


class RegularGridsConvolutionalProcess(ConvolutionalProcess):
    """
    Special case of a Convolutional Process [Jonathan] when the input, output and
    pseudo points are on a grid of the same size.

    Notes
    -----
    - Assumes that input, output and pseudo points are on the same grid
    - Assumes that Y_cntxt is the the grid values (y_dim / channels on last dim),
    while X_cntxt and X_trgt are confidence masks of the shape of the grid rather
    than set of features.
    - This cannot be used for sub-pixel interpolation / super resolution.
    - As X_cntxt is a grid, each batch example could have a different number of
    contexts (i.e. different number of non zeros). X_trgt still has to have the
    same amount of non zero due to how it is computed.
    - As we do not use a set convolution, the receptive field is easy to specify,
    making the model much more computationally efficient.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    Conv : nn.Module, optional
        Convolution to use for the first normalized convolution.

    PseudoTransformer : callable, optional
        Object used to transform the pseudo inputs {pseudo_inp} -> {pseudo_inp}. Note
        that the temporary queries will be uniformly sampled and you can thus use
        a convolution instead. Example:
            - `partial(CNN, is_chan_last=True)` : uses a multilayer CNN. To
            be compatible with self attention the channel layer should be last.
            - `SelfAttention` : uses a self attention layer.

    kwargs :
        Additional arguments to `ConvolutionalProcess`.

    References
    ----------
    [Jonathan]
    """

    def __init__(self, x_dim, y_dim,
                 # uses only depth wise + make sure positive to be interpreted as a density
                 Conv=lambda y_dim: make_abs_conv(nn.Conv2d)(y_dim, y_dim, groups=y_dim,
                                                             kernel_size=19, padding=19 // 2,
                                                             bias=False),
                 PseudoSelfAttn=partial(CNN, ResConvBlock,
                                        Conv=nn.Conv2d,
                                        n_blocks=3,
                                        Normalization=nn.Identity,
                                        is_chan_last=True,
                                        kernel_size=11),
                 **kwargs):
        super().__init__(keys_to_pseudo=lambda *args, **kwargs: None,  # will redefine
                         PseudoSelfAttn=PseudoSelfAttn,
                         pseudo_to_queries=lambda *args, **kwargs: None)  # will redefine

        self.conv = Conv(y_dim)
        self.resizer = nn.Linear(self.y_dim * 2, self.r_dim)  # 2 because also confidence channels
        self.density_to_conf = ProbabilityConverter(is_train_temperature=True,
                                                    is_train_bias=True,
                                                    trainable_dim=self.y_dim,
                                                    # higher density => higher conf
                                                    temperature_transformer=torch.softplus)

        self.tmp_self_attender = TmpSelfAttn(self.r_dim)

        self.reset_parameters()

    def keys_to_pseudo(self, mask_context, _, X):
        batch_size, *grid_shape, y_dim = X.shape

        # channels have to be in second dimension for convolution
        X = channels_to_2nd_dim(X)
        mask_context = channels_to_2nd_dim(mask_context).float()

        # a * (c f)
        numerator = self.conv(X * mask_context)
        # a * c
        denominator = self.conv(mask_context.expand_as(X))
        # normalized convolution
        out = numerator / torch.clamp(denominator, min=1e-5)
        # initial density could be very large => make sure not saturating sigmoid (*0.1)
        confidence = self.density_to_conf(denominator.view(-1, self.y_dim) * 0.1
                                          ).view(batch_size, self.y_dim, *grid_shape)
        # don't concatenate density but a bounded version ("confidence") =>
        # doesn't break under high density
        out = torch.cat([out, confidence], dim=1)

        out = self.resizer(channel_to_last_dim(out))

        return out

    def pseudo_to_queries(self, _, __, pseudo_values):
        """Return the pseudo values a they are on the same grid as the querries."""
        return pseudo_values

    def make_dec_inp(self, X, _, mask_target):
        """Make the context input for the decoder."""
        batch_size, *grid_shape, y_dim = X.shape

        # only works because number of targets same in all batch
        linear_trgts = X.masked_select(mask_target).view(batch_size, -1, y_dim)

        return linear_trgts
