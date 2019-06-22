import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.independent import Independent
from torch.distributions import Normal


from neuralproc.utils.initialization import weights_init
from neuralproc.utils.torchextend import min_max_scale, MultivariateNormalDiag, MLP
from neuralproc.utils.datasplit import CntxtTrgtGetter
from neuralproc.utils.attention import get_attender
from .encoders import get_uninitialized_mlp, merge_flat_input


__all__ = ["NeuralProcess", "AttentiveNeuralProcess"]


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
        Dimension of the encoded X. If `None` uses `r_dim`.

    XEncoder : nn.Module, optional
        Spatial encoder module which maps {x_i} -> x_transf_i. It should be
        constructable via `xencoder(x_dim, x_transf_dim)`. Example:
            - `MLP` : will learn positional embeddings with MLP
            - `SinusoidalEncodings` : use sinusoidal positional encodings.

    XYEncoder : nn.Module, optional
        Encoder module which maps {x_transf_i, y_i} -> {r_i}. It should be constructable
        via `xyencoder(x_transf_dim, y_dim, n_out)`. If you have an encoder that maps
        xy -> r you can convert it via `merge_flat_input(Encoder)`. Example:
            - `merge_flat_input(MLP, is_sum_merge=False)` : learn representation
            with MLP. `merge_flat_input` concatenates (or sums) X and Y inputs.
            - `SelfAttentionBlock` : self attention mechanisms as [4]. For more parameters
            (attention type, number of layers ...) refer to its docstrings.

    Decoder : nn.Module, optional
        Decoder module which maps {x_t, r} -> {y_hat_t}. It should be constructable
        via `decoder(x, r_dim, n_out)`. If you have an decoder that maps
        rx -> y you can convert it via `merge_flat_input(Decoder)`. Example:
            - `merge_flat_input(MLP)` : predict with MLP.
            - `SelfAttentionBlock` : predict with self attention mechanisms to
            have coherant predictions (not use in attentive neural process [4] but
            in image transformer [5]).

    aggregator : callable, optional
        Agregreator function which maps {r_i} -> r. It should have a an argument
        `dim` to say specify the dimensions of aggregation. The dimension should
        not be kept (i.e. keepdim=False). To use a cross attention aggregation,
        use `AttentiveNeuralProcess` instead of `NeuralProcess`.

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suff_stat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`. Only used if `encoded_path in ["latent",
        "both"]`.

    get_cntxt_trgt : callable, optional
        Function that split the input into context and target points.
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, y, **kwargs)`.
        Note: context points should be a subset of target ones. If you already
        have the context and target point, put them in a dictionary and split
        the dictionary in `get_cntxt_trgt`.

    encoded_path : {"deterministic", "latent", "both"}
        Which path(s) to use:
        - `"deterministic"` uses a Conditional Neural Process [2] (no latents),
        where the decoder gets a deterministic representation as input
        (function of the context).
        - `"latent"` uses the original Neural Process [1], where the decoder gets
        a sample latent representation as input (function of the target during
        training and context during test).
        If `"both"` concatenates both representations as described in [4].

    PredictiveDistribution : torch.distributions.Distribution, optional
        Predictive distribution. The predicted outputs will be independent and thus
        wrapped around `Independent` (e.g. diagonal covariance for a Gaussian).
        The input to the constructor are currently a value in ]-inf, inf[ and one
        in [0.1, inf[ (typically `loc` and `scale`), although it is very easy to make
        more general if needs be.

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
                 x_transf_dim=None,
                 XEncoder=MLP,
                 XYEncoder=merge_flat_input(get_uninitialized_mlp(n_hidden_layers=2),
                                            is_sum_merge=True),
                 Decoder=merge_flat_input(get_uninitialized_mlp(n_hidden_layers=4),
                                          is_sum_merge=True),
                 aggregator=torch.mean,
                 LatentEncoder=MLP,
                 get_cntxt_trgt=CntxtTrgtGetter(is_add_cntxts_to_trgts=True),
                 encoded_path="deterministic",
                 PredictiveDistribution=Normal):
        super().__init__()
        self.get_cntxt_trgt = get_cntxt_trgt
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path.lower()
        self.PredictiveDistribution = PredictiveDistribution
        self.x_transf_dim = x_transf_dim if x_transf_dim is not None else self.r_dim

        self.x_encoder = XEncoder(self.x_dim, self.x_transf_dim)
        self.xy_encoder = XYEncoder(self.x_transf_dim, self.y_dim, self.r_dim)
        self.aggregator = aggregator
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

    def forward(self, X, y=None, **kwargs):
        """
        Split context and target in the class to make it compatible with
        usual datasets and training frameworks, then redirects to `forward_step`.
        """
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, y, **kwargs)
        return self.forward_step(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    def forward_step(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
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
            Set of all target features {x_t}. Note: context points should be a
            subset of target ones.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution
            Target distribution.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Set of all target values {y_t}, returned to redirect it to the loss
            function.

        q_z_trgt: torch.distributions.Distribution
            Latent distribution for the targets. `None` if `LatentEncoder=None`
            or not training.

        q_z_cntxt: torch.distributions.Distribution
            Latent distribution for the context points. `None` if
            `LatentEncoder=None` or not training.
        """
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

        return p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt

    def latent_path(self, X, Y):
        """Latent encoding path."""
        # batch_size, n_cntxt, r_dim
        X_transf = self.x_encoder(X)
        R_cntxt = self.xy_encoder(X_transf, Y)
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)

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
        # batch_size, n_cntxt, r_dim
        X_transf = self.x_encoder(X_cntxt)
        R_cntxt = self.xy_encoder(X_transf, Y_cntxt)
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)

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
        """
        X_transf = self.x_encoder(X_trgt)
        # batch_size, n_trgt, y_dim*2
        suff_stat_Y_trgt = self.decoder(X_transf, dec_inp)
        loc_trgt, scale_trgt = suff_stat_Y_trgt.split(self.y_dim, dim=-1)
        # Following convention "Empirical Evaluation of Neural Process Objectives"
        scale_trgt = 0.1 + 0.9 * F.softplus(scale_trgt)
        p_y = Independent(self.PredictiveDistribution(loc_trgt, scale_trgt), 1)
        return p_y


class AttentiveNeuralProcess(NeuralProcess):
    """
    Wrapper around `NeuralProcess` that implements an attentive neural process [4].

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    attention : {'multiplicative', "additive", "scaledot", "multihead", "manhattan",
                "euclidean", "cosine"}, optional
        Type of attention to use. More details in `get_attender`.

    is_normalize : bool, optional
        Whether attention weights should sum to 1 (using softmax). If not weights
        will be in [0,1] but not necessarily sum to 1.

    kwargs :
        Additional arguments to `NeuralProcess`.

    References
    ----------
    [4] Kim, Hyunjik, et al. "Attentive neural processes." arXiv preprint
        arXiv:1901.05761 (2019).
    """

    def __init__(self, x_dim, y_dim,
                 attention="scaledot",
                 is_normalize=True,
                 encoded_path="both",
                 **kwargs):

        super().__init__(x_dim, y_dim, encoded_path=encoded_path, **kwargs)

        self.attender = get_attender(attention, self.x_transf_dim,
                                     is_normalize=is_normalize, **kwargs)

        self.reset_parameters()

    def deterministic_path(self, X_cntxt, Y_cntxt, X_trgt):
        """
        Deterministic encoding path.
        """
        # batch_size, n_cntxt, r_dim
        keys = self.x_encoder(X_cntxt)
        values = self.xy_encoder(keys, Y_cntxt)

        # batch_size, n_n_trgt, r_dim
        queries = self.x_encoder(X_trgt)

        # batch_size, n_trgt, value_size
        R_attn = self.attender(keys, queries, values)
        return R_attn


class ConvolutionNeuralProcess(nn.Module):
    """
    Implements a convolution neural process.

    Parameters
    ----------
    x_dim : int
        Dimension of features.

    y_dim : int
        Dimension of y values.

    r_dim : int, optional
        Dimension of representation.

    x_transf_dim : int, optonal
        Dimension of the encoded X. If `None` uses `r_dim`.

    XEncoder : nn.Module, optional
        Spatial encoder module which maps {x_i} -> x_transf_i. It should be
        constructable via `xencoder(x_dim, x_transf_dim)`. Example:
            - `MLP` : will learn positional embeddings with MLP
            - `SinusoidalEncodings` : use sinusoidal positional encodings.

    XYEncoder : nn.Module, optional
        Encoder module which maps {x_transf_i, y_i} -> {r_i}. It should be constructable
        via `xyencoder(x_transf_dim, y_dim, n_out)`. If you have an encoder that maps
        xy -> r you can convert it via `merge_flat_input(Encoder)`. Example:
            - `merge_flat_input(MLP, is_sum_merge=False)` : learn representation
            with MLP. `merge_flat_input` concatenates (or sums) X and Y inputs.
            - `SelfAttentionBlock` : self attention mechanisms as [4]. For more parameters
            (attention type, number of layers ...) refer to its docstrings.

    Decoder : nn.Module, optional
        Decoder module which maps {x_t, r} -> {y_hat_t}. It should be constructable
        via `decoder(x, r_dim, n_out)`. If you have an decoder that maps
        rx -> y you can convert it via `merge_flat_input(Decoder)`. Example:
            - `merge_flat_input(MLP)` : predict with MLP.
            - `SelfAttentionBlock` : predict with self attention mechanisms to
            have coherant predictions (not use in attentive neural process [4] but
            in image transformer [5]).

    aggregator : callable, optional
        Agregreator function which maps {r_i} -> r. It should have a an argument
        `dim` to say specify the dimensions of aggregation. The dimension should
        not be kept (i.e. keepdim=False). To use a cross attention aggregation,
        use `AttentiveNeuralProcess` instead of `NeuralProcess`.

    LatentEncoder : nn.Module, optional
        Encoder which maps r -> z_suff_stat. It should be constructed via
        `LatentEncoder(r_dim, n_out)`. Only used if `encoded_path in ["latent",
        "both"]`.

    split_cntxt_trgt : callable, optional
        Function that split the input into context and target points.
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, **kwargs).
        Note: context points should be a subset of target ones. If you already
        have the context and target point, put them in a dictionary and split
        the dictionary in `get_cntxt_trgt`.

    encoded_path : {"deterministic", "latent", "both"}
        Which path(s) to use:
        - `"deterministic"` uses a Conditional Neural Process [2] (no latents),
        where the decoder gets a deterministic representation as input
        (function of the context).
        - `"latent"` uses the original Neural Process [1], where the decoder gets
        a sample latent representation as input (function of the target during
        training and context during test).
        If `"both"` concatenates both representations as described in [4].

    PredictiveDistribution : torch.distributions.Distribution, optional
        Predictive distribution. The predicted outputs will be independent and thus
        wrapped around `Independent` (e.g. diagonal covariance for a Gaussian).
        The input to the constructor are currently a value in ]-inf, inf[ and one
        in [0.1, inf[ (typically `loc` and `scale`), although it is very easy to make
        more general if needs be.
    """

    def __init__(self, x_dim, y_dim,
                 Decoder=merge_flat_input(get_uninitialized_mlp(n_hidden_layers=4),
                                          is_sum_merge=True),
                 get_cntxt_trgt=CntxtTrgtGetter(is_add_cntxts_to_trgts=False),
                 PredictiveDistribution=Normal):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.encoded_path = encoded_path.lower()
        self.PredictiveDistribution = PredictiveDistribution
        self.x_transf_dim = x_transf_dim if x_transf_dim is not None else self.r_dim

        self.x_encoder = XEncoder(self.x_dim, self.x_transf_dim)
        self.xy_encoder = XYEncoder(self.x_transf_dim, self.y_dim, self.r_dim)
        self.aggregator = aggregator
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

    def forward(self, X, y=None, **kwargs):
        """
        Split context and target in the class to make it compatible with
        usual datasets and training frameworks, then redirects to `forward_step`.
        """
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = self.get_cntxt_trgt(X, y, **kwargs)
        return self.forward_step(X_cntxt, Y_cntxt, X_trgt, Y_trgt=Y_trgt)

    def forward_step(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
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
            Set of all target features {x_t}. Note: context points should be a
            subset of target ones.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution
            Target distribution.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Set of all target values {y_t}, returned to redirect it to the loss
            function.

        q_z_trgt: torch.distributions.Distribution
            Latent distribution for the targets. `None` if `LatentEncoder=None`
            or not training.

        q_z_cntxt: torch.distributions.Distribution
            Latent distribution for the context points. `None` if
            `LatentEncoder=None` or not training.
        """
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

        return p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt

    def latent_path(self, X, Y):
        """Latent encoding path."""
        # batch_size, n_cntxt, r_dim
        X_transf = self.x_encoder(X)
        R_cntxt = self.xy_encoder(X_transf, Y)
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)

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
        # batch_size, n_cntxt, r_dim
        X_transf = self.x_encoder(X_cntxt)
        R_cntxt = self.xy_encoder(X_transf, Y_cntxt)
        # batch_size, r_dim
        r = self.aggregator(R_cntxt, dim=1)

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
        """
        X_transf = self.x_encoder(X_trgt)
        # batch_size, n_trgt, y_dim*2
        suff_stat_Y_trgt = self.decoder(X_transf, dec_inp)
        loc_trgt, scale_trgt = suff_stat_Y_trgt.split(self.y_dim, dim=-1)
        # Following convention "Empirical Evaluation of Neural Process Objectives"
        scale_trgt = 0.1 + 0.9 * F.softplus(scale_trgt)
        p_y = Independent(self.PredictiveDistribution(loc_trgt, scale_trgt), 1)
        return p_y
