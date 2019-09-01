import random

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from neuralproc.utils.helpers import prod, channels_to_2nd_dim
from utils.data import cntxt_trgt_collate
from utils.predict import VanillaPredictor
from .visualize_1d import _get_p_y_pred

__all__ = ["plot_dataset_samples_imgs", "plot_posterior_img"]

DFLT_FIGSIZE = (17, 9)


def plot_dataset_samples_imgs(dataset, n_plots=4, figsize=DFLT_FIGSIZE, ax=None,
                              pad_value=1):
    """Plot `n_samples` samples of the a datset."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img_tensor = torch.stack([dataset[random.randint(0, len(dataset) - 1)][0]
                              for i in range(n_plots)], dim=0)
    grid = make_grid(img_tensor,
                     nrow=2,
                     pad_value=pad_value)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')


def plot_posterior_img(data, get_cntxt_trgt, model,
                       MeanPredictor=VanillaPredictor,
                       is_uniform_grid=True,
                       n_plots=4,
                       figsize=(18, 4),
                       ax=None):
    """
    Plot the mean of the estimated posterior for images.

    Parameters
    ----------
    data : Dataset
        Dataset from which to sample the images.

    get_cntxt_trgt : callable
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

    model : nn.Module
        Model used to initialize `MeanPredictor`.

    MeanPredictor : untitialized callable, optional
        Callable which is initalized with `MeanPredictor(model)` and then takes as
        input `X_cntxt, Y_cntxt, X_trgt` (`mask_cntxt, X, mask_trgt` if
        `is_uniform_grid`) and returns the mean the posterior. E.g. `VanillaPredictor`
        or `AutoregressivePredictor`.

    is_uniform_grid : bool, optional
        Whether the input are the image and corresponding masks rather than
        the slected pixels. Typically used for `RegularGridsConvolutionalProcess`.

    n_plots : int, optional
        Number of images to samples. They will be plotted in different columns.

    figsize : tuple, optional

    ax : plt.axes.Axes, optional
    """

    model.eval()

    dim_grid = 2 if is_uniform_grid else 1
    imgs = [data[random.randint(0, len(data))] for _ in range(n_plots)]
    cntxt_trgt = cntxt_trgt_collate(get_cntxt_trgt, is_return_masks=is_uniform_grid)(imgs)[0]
    mask_cntxt, X, mask_trgt, _ = (cntxt_trgt["X_cntxt"], cntxt_trgt["Y_cntxt"],
                                   cntxt_trgt["X_trgt"], cntxt_trgt["Y_trgt"])
    mean_y = MeanPredictor(model)(mask_cntxt, X, mask_trgt)

    if is_uniform_grid:
        mean_y = mean_y.view(*X.shape)

    if X.shape[-1] == 1:
        X = X.expand(-1, *[-1] * dim_grid, 3)
        mean_y = mean_y.expand(-1, *[-1] * dim_grid, 3)

    if is_uniform_grid:
        background = data.missing_px_color.view(1, *[1] * dim_grid, 3
                                                ).expand(*mean_y.shape).clone()
        out_cntxt = torch.where(mask_cntxt, X, background)

        background[mask_trgt.squeeze(-1)] = mean_y.view(-1, 3)
        out_pred = background.clone()

    else:
        out_cntxt = points_to_grid(mask_cntxt,
                                   X,
                                   data.shape[1:],
                                   background=data.missing_px_color)

        out_pred = points_to_grid(mask_trgt,
                                  mean_y,
                                  data.shape[1:],
                                  background=data.missing_px_color)

    outs = [out_cntxt, out_pred]

    grid = make_grid(channels_to_2nd_dim(torch.cat(outs, dim=0)),
                     nrow=n_plots,
                     pad_value=1.)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(grid.permute(1, 2, 0).numpy(), )
    ax.axis('off')

    return grid


def idcs_grid_to_idcs_flatten(idcs, grid_shape):
    """Convert a tensor containing indices of a grid to indices on the flatten grid."""
    for i, size in enumerate(grid_shape):
        idcs[:, :, i] *= prod(grid_shape[i + 1:])
    return idcs.sum(-1)


def points_to_grid(X, Y, grid_shape, background=torch.tensor([0., 0., 0.])):
    """Converts points to a grid (undo mask select from datasplit)"""

    batch_size, _, y_dim = Y.shape
    X = X.clone()
    background = background.view(1, *(1 for _ in grid_shape), y_dim).repeat(batch_size, *grid_shape, 1)

    for i, size in enumerate(grid_shape):
        X[:, :, i] += 1  # in [0,2]
        X[:, :, i] /= 2 / (size - 1)  # in [0,size]

    X = X.long()
    X = idcs_grid_to_idcs_flatten(X, grid_shape)

    background = background.view(batch_size, -1, y_dim)

    for b in range(batch_size):
        background[b, X[b], :] = Y[b]

    background = background.view(batch_size, *grid_shape, y_dim)

    return background
