

__all__ = ["VanillaPredictor", "AutoregressivePredictor", "get_next_autoregressive_pixel",
           "get_next_autoregressive_axis", "AutoregressivePredictor"]


class VanillaPredictor:
    """Single shot prediction using a trained `NeuralProcess` model."""

    def __init__(self, model):
        self.model = model

    def __call__(self, *args):
        p_y_pred, *_ = self.model(*args)
        mean_y = p_y_pred.base_dist.loc.detach()
        return mean_y


def get_next_autoregressive_pixel(mask_cntxt, mask_trgt, axis=1):
    """
    Given the current context mask and the final target mask, return the next
    temporary target mask by setting one pixel by pixel (iterating over the given
    axis)
    """
    if not (mask_trgt == 1).all():
        raise NotImplementedError("Currently `AutoregressivePredictor` only works for whole image predictions.")

    next_mask_cntxt = mask_cntxt.clone()
    slcs = [slice(None)] * (len(mask_cntxt.shape))

    # could use generator instead of looping from the begining
    for i in range(next_mask_cntxt.shape[axis]):
        slcs[axis + 1] = i  # axis doesn't take into account batch => increment
        for j in range(next_mask_cntxt[slcs].shape[1]):
            if not (next_mask_cntxt[slcs][:, j, ...] == 1).all():
                next_mask_cntxt[slcs][:, j, ...] = 1
                return next_mask_cntxt

    return next_mask_cntxt


def get_next_autoregressive_axis(mask_cntxt, mask_trgt, axis=1):
    """
    Given the current context mask and the final target mask, return the next
    temporary target mask by setting all the next axis (e.g. columns `for axis=1`)
    to ones.
    """
    if not (mask_trgt == 1).all():
        raise NotImplementedError("Currently `AutoregressivePredictor` only works for whole image predictions.")

    next_mask_cntxt = mask_cntxt.clone()
    slcs = [slice(None)] * (len(mask_cntxt.shape))

    # could use generator instead of looping from the begining
    for i in range(next_mask_cntxt.shape[axis]):
        slcs[axis + 1] = i  # axis doesn't take into account batch => increment
        if not (next_mask_cntxt[slcs] == 1).all():
            next_mask_cntxt[slcs] = 1
            break

    return next_mask_cntxt


class AutoregressivePredictor:
    """
    Autoregressive prediction using a trained `NeuralProcess` model.

    Parameters
    ----------
    model : nn.Module
        Model used to initialize `MeanPredictor`.

    get_next_tgrts : callable, optional
        Function which returns the next mask target given the current
        mask context and the final mask target
        `get_next_tgrts(mask_cntxt, mask_trgt)`.

    is_repredict : bool, optional
        Whether to is_repredict the given context and previous targets at
        each autoregressive temporary steps.
    """

    def __init__(self, model,
                 get_next_tgrts=get_next_autoregressive_axis,
                 is_repredict=False):
        self.predictor = VanillaPredictor(model)
        self.get_next_tgrts = get_next_tgrts
        self.is_repredict = is_repredict

    def __call__(self, mask_cntxt, X, mask_trgt):
        X = X.clone()

        while not (mask_cntxt == 1).all():

            cur_mask_trgt = self.get_next_tgrts(mask_cntxt, mask_trgt)
            next_mask_cntxt = cur_mask_trgt.clone()

            if not self.is_repredict:
                # don't predict what is in cntxt
                cur_mask_trgt[mask_cntxt.squeeze(-1)] = 0

            mean_y = self.predictor(mask_cntxt, X, cur_mask_trgt)
            X[cur_mask_trgt.squeeze(-1)] = mean_y.view(-1, mean_y.shape[-1])
            mask_cntxt = next_mask_cntxt

        # predict once with all
        mean_y = self.predictor(mask_trgt, X, mask_trgt)

        return mean_y
