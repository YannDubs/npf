import torch

__all__ = ["VanillaPredictor", "AutoregressivePredictor",
           "get_next_autoregressive_closest_pixels"]


class VanillaPredictor:
    """Single shot prediction using a trained `NeuralProcess` model."""

    def __init__(self, model):
        self.model = model

    def __call__(self, *args):
        p_y_pred, *_ = self.model(*args)
        mean_y = p_y_pred.base_dist.loc.detach()
        return mean_y


def get_shifted_masks(mask):
    """Given a batch of masks , returns the same masks shifted to the right,
    left, up, down."""
    right_shifted = torch.cat((mask[:, :, -1:, ...] * 0, mask[:, :, :-1, ...]), dim=2)
    left_shifted = torch.cat((mask[:, :, 1:, ...], mask[:, :, :1, ...] * 0), dim=2)
    up_shifted = torch.cat((mask[:, 1:, :, ...], mask[:, :1, :, ...] * 0), dim=1)
    down_shifted = torch.cat((mask[:, -1:, :, ...] * 0, mask[:, :-1, :, ...]), dim=1)
    return right_shifted, left_shifted, up_shifted, down_shifted


def get_next_autoregressive_closest_pixels(mask_cntxt):
    """
    Given the current context mask, return the next
    temporary target mask by setting all pixels than are at 1 manhatan distance
    of a context pixel.
    """
    next_mask_cntxt = mask_cntxt.clone()
    slcs = [slice(None)] * (len(mask_cntxt.shape))

    while not (next_mask_cntxt == 1).all():
        # shift array to the 4 directions to get all neighbours
        right, left, up, down = get_shifted_masks(next_mask_cntxt)
        # set all neigbours to 1 by summing + make sure nothing over 1
        next_mask_cntxt = torch.clamp(right + left + up + down + next_mask_cntxt, 0, 1)
        yield next_mask_cntxt.clone()


class AutoregressivePredictor:
    """
    Autoregressive prediction using a trained `NeuralProcess` model.

    Parameters
    ----------
    model : nn.Module
        Model used to initialize `MeanPredictor`.

    gen_autoregressive_trgts : callable, optional
        Function which returns a generator of the next mask target given the inital
        mask context `get_next_tgrts(mask_cntxt)`.

    is_repredict : bool, optional
        Whether to is_repredict the given context and previous targets at
        each autoregressive temporary steps.
    """

    def __init__(self, model,
                 gen_autoregressive_trgts=get_next_autoregressive_closest_pixels,
                 is_repredict=False):
        self.predictor = VanillaPredictor(model)
        self.gen_autoregressive_trgts = gen_autoregressive_trgts
        self.is_repredict = is_repredict

    def __call__(self, mask_cntxt, X, mask_trgt):
        X = X.clone()

        gen_cur_mask_trgt = self.gen_autoregressive_trgts(mask_cntxt)

        for cur_mask_trgt in gen_cur_mask_trgt:
            next_mask_cntxt = cur_mask_trgt.clone()

            if not self.is_repredict:
                # don't predict what is in cntxt
                cur_mask_trgt[mask_cntxt.squeeze(-1)] = 0

            mean_y = self.predictor(mask_cntxt, X, cur_mask_trgt)
            X[cur_mask_trgt.squeeze(-1)] = mean_y.view(-1, mean_y.shape[-1])
            mask_cntxt = next_mask_cntxt

        # predict once with all to have the actual trgt
        mean_y = self.predictor(mask_trgt, X, mask_trgt)

        return mean_y
