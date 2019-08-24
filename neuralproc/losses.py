import torch.nn as nn
from torch.distributions.kl import kl_divergence

__all__ = ["NeuralProcessLoss"]


class NeuralProcessLoss(nn.Module):
    """
    Compute the Neural Process Loss [1] but unbiased [Jonathan].

    Parameters
    ----------
    get_beta : callable, optional
        Function which returns the weight of the kl divergence given `is_training`.

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    [Jonathan]
    """

    def __init__(self, get_beta=lambda _: 1):
        super().__init__()
        self.get_beta = get_beta

    def forward(self, pred_outputs, Y_trgt, weight=None):
        """Compute the Neural Process Loss averaged over the batch.

        Parameters
        ----------
        pred_outputs : tuple
            Tuple of (p_y_trgt, q_z_trgt, q_z_cntxt). Output of NeuralProcess.

        Y_trgt: torch.Tensor, size=[batch_size, n_trgt, y_dim]
            Set of all target values {y_t}.

        weight : torch.Tensor, size = [batch_size,]
            Weight of every example. If None, every example is weighted by 1.
        """
        p_y_trgt, q_z_trgt, q_z_cntxt = pred_outputs
        batch_size, n_trgt, _ = Y_trgt.shape

        # mean over all targets => unbiased estimate of the autoregressive loss
        neg_log_like = - p_y_trgt.log_prob(Y_trgt).view(batch_size, -1).mean(-1)

        if q_z_trgt is not None:
            # during validation the kl will be 0 because we do not compute q_z_trgt
            kl_loss = kl_divergence(q_z_trgt, q_z_cntxt)
            # kl is multivariate Gaussian => sum over all target but we want mean
            kl_loss = kl_loss / n_trgt
        else:
            kl_loss = 0

        loss = neg_log_like + self.get_beta(self.training) * kl_loss

        if weight is not None:
            loss = loss * weight

        return loss.mean(dim=0)
