import torch.nn as nn
from torch.distributions.kl import kl_divergence

__all__ = ["NeuralProcessLoss"]


class NeuralProcessLoss(nn.Module):
    """
    Compute the Neural Process Loss [1].

    Parameters
    ----------
    get_beta : callable, optional
        Function which returns the weight of the kl divergence given `is_training`.

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def __init__(self, get_beta=lambda _: 1):
        super().__init__()
        self.get_beta = get_beta
        self.printed = False

    def forward(self, inputs, y=None, weight=None):
        """Compute the Neural Process Loss averaged over the batch.

        Parameters
        ----------
        inputs: tuple
            Tuple of (p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt). This can directly
            take the output of NueralProcess.

        y: None
            Placeholder.

        weight: torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        p_y_trgt, Y_trgt, q_z_trgt, q_z_cntxt, summary = inputs
        batch_size, n_trgt, _ = Y_trgt.shape

        neg_log_like = - p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).mean(-1)

        if q_z_trgt is not None:
            # use latent variables and training
            # note that during validation the kl will actually be 0 because
            # we do not compute q_z_trgt
            kl_loss = kl_divergence(q_z_trgt, q_z_cntxt)
            # kl is multivariate Gaussian => sum over all target but we want mean
            kl_loss = kl_loss / n_trgt
        else:
            kl_loss = 0

        if summary is not None:
            summary = summary.view(batch_size, -1)
            deltas = summary - summary.mean(0, keepdim=True)
            deltas = deltas.abs().mean()
        else:
            deltas = 0

        loss = neg_log_like + self.get_beta(self.training) * kl_loss + deltas

        if weight is not None:
            loss = loss * weight

        return loss.mean(dim=0)
