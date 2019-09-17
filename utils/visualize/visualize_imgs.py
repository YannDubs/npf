import random
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np
from skorch.dataset import unpack_data

from neuralproc.utils.helpers import prod, channels_to_2nd_dim
from neuralproc.utils.predict import VanillaPredictor
from utils.data import cntxt_trgt_collate
from utils.helpers import set_seed, tuple_cont_to_cont_tuple
from utils.train import EVAL_FILENAME
from .visualize_1d import _get_p_y_pred

__all__ = ["plot_dataset_samples_imgs", "plot_posterior_img", "plot_qualitative_with_kde"]

DFLT_FIGSIZE = (17, 9)


def plot_dataset_samples_imgs(
    dataset, n_plots=4, figsize=DFLT_FIGSIZE, ax=None, pad_value=1, seed=123
):
    """Plot `n_samples` samples of the a datset."""
    set_seed(seed)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img_tensor = torch.stack(
        [dataset[random.randint(0, len(dataset) - 1)][0] for i in range(n_plots)], dim=0
    )
    grid = make_grid(img_tensor, nrow=2, pad_value=pad_value)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")


import seaborn as sns


def plot_posterior_img(
    data,
    get_cntxt_trgt,
    model,
    MeanPredictor=VanillaPredictor,
    is_uniform_grid=True,
    img_indcs=None,
    n_plots=4,
    figsize=(18, 4),
    ax=None,
    seed=123,
    is_return=False,
):  # TO DOC
    """
    Plot the mean of the estimated posterior for images.

    Parameters
    ----------
    data : Dataset
        Dataset from which to sample the images.

    get_cntxt_trgt : callable or dict
        Function that takes as input the features and tagrets `X`, `y` and return
        the corresponding `X_cntxt, Y_cntxt, X_trgt, Y_trgt`. If dict should contain the correct 
        `X_cntxt, Y_cntxt, X_trgt, Y_trgt`.

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

    img_indcs : list of int, optional
        Indices of the images to plot. If `None` will randomly sample `n_plots`
        of them.

    n_plots : int, optional
        Number of images to samples. They will be plotted in different columns.
        Only used if `img_indcs` is `None`.

    figsize : tuple, optional

    ax : plt.axes.Axes, optional

    seed : int, optional
    """
    set_seed(seed)

    model.eval()

    dim_grid = 2 if is_uniform_grid else 1
    if isinstance(get_cntxt_trgt, dict):
        mask_cntxt, X, mask_trgt, _ = (
            get_cntxt_trgt["X_cntxt"],
            get_cntxt_trgt["Y_cntxt"],
            get_cntxt_trgt["X_trgt"],
            get_cntxt_trgt["Y_trgt"],
        )
        n_plots = mask_cntxt.size(0)

    else:
        if img_indcs is None:
            img_indcs = [random.randint(0, len(data)) for _ in range(n_plots)]
        n_plots = len(img_indcs)
        imgs = [data[i] for i in img_indcs]

        cntxt_trgt = cntxt_trgt_collate(get_cntxt_trgt, is_return_masks=is_uniform_grid)(imgs)[0]
        mask_cntxt, X, mask_trgt, _ = (
            cntxt_trgt["X_cntxt"],
            cntxt_trgt["Y_cntxt"],
            cntxt_trgt["X_trgt"],
            cntxt_trgt["Y_trgt"],
        )

    mean_y = MeanPredictor(model)(mask_cntxt, X, mask_trgt)

    if is_uniform_grid:
        mean_y = mean_y.view(*X.shape)

    if X.shape[-1] == 1:
        X = X.expand(-1, *[-1] * dim_grid, 3)
        mean_y = mean_y.expand(-1, *[-1] * dim_grid, 3)

    if is_uniform_grid:
        background = (
            data.missing_px_color.view(1, *[1] * dim_grid, 3).expand(*mean_y.shape).clone()
        )
        out_cntxt = torch.where(mask_cntxt, X, background)

        background[mask_trgt.squeeze(-1)] = mean_y.view(-1, 3)
        out_pred = background.clone()

    else:
        out_cntxt = points_to_grid(mask_cntxt, X, data.shape[1:], background=data.missing_px_color)

        out_pred = points_to_grid(
            mask_trgt, mean_y, data.shape[1:], background=data.missing_px_color
        )

    outs = [out_cntxt, out_pred]

    grid = make_grid(channels_to_2nd_dim(torch.cat(outs, dim=0)), nrow=n_plots, pad_value=1.0)

    if is_return:
        return grid

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")


def plot_qualitative_with_kde(
    trainer,
    dataset,
    trainer_compare=None,
    n_images=8,
    figsize=DFLT_FIGSIZE,
    title=None,
    seed=123,
    **kwargs
):
    """
    Plot qualitative samples using `plot_posterior_img` but select the samples and mask to plot
    given the score at test time. 
    """

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    img_indcs = []

    trainers = [trainer]
    if trainer_compare is not None:
        trainers.append(trainer_compare)

    for name, trainer in trainers:
        chckpnt_dirname = dict(trainer.callbacks_)["Checkpoint"].dirname
        test_eval_file = os.path.join(chckpnt_dirname, EVAL_FILENAME)
        test_loglike = np.loadtxt(test_eval_file, delimiter=",")
        sns.kdeplot(test_loglike, ax=ax, shade=True, label=name)
        sns.despine()

        # find indices such that same space between all
        values = np.linspace(test_loglike.min(), test_loglike.max(), n_images)
        idcs = [(np.abs(test_loglike - v)).argmin() for v in values]
        sorted_idcs = list(np.sort(idcs))[::-1]
        selected_data = []

        set_seed(seed)  # make sure same order and indices for cntxt and trgt

        trainer.notify("on_epoch_begin", dataset_valid=dataset)
        i = 0
        next_idx = sorted_idcs.pop()
        for data in trainer.get_iterator(dataset, training=False):
            Xi, yi = unpack_data(data)

            if next_idx < i + yi.size(0):
                cur_idx = next_idx - i
                selected_data.append(
                    {k: v[cur_idx : cur_idx + 1, ...] for k, v in data[0].items()}
                )
                next_idx = sorted_idcs.pop()
            else:
                i += yi.size(0)
                continue

        selected_data = {
            k: torch.cat(v, dim=0) for k, v in tuple_cont_to_cont_tuple(selected_data).items()
        }

    ax.legend()

    if title is not None:
        ax.set_title(title, fontsize=18)

    # ---- TMP : USE BETTER WAY OF DOING -----
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[0]))

    grids = []
    for name, trainer in trainers.items():
        grids.append(
            plot_posterior_img(
                data,
                get_cntxt_trgt,
                trainer.module_,
                img_indcs=img_indcs,
                is_uniform_grid="grided" in name.lower(),  # use a better way # DEV MODE
                is_return=True,
                **kwargs
            )
        )

    if len(grids) > 1:
        grid = torch.cat((grids[0], grids[1][:, 36:, :]), dim=1)
    else:
        grid = grids[0]

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")
    # ----------------------------------------


def idcs_grid_to_idcs_flatten(idcs, grid_shape):
    """Convert a tensor containing indices of a grid to indices on the flatten grid."""
    for i, size in enumerate(grid_shape):
        idcs[:, :, i] *= prod(grid_shape[i + 1 :])
    return idcs.sum(-1)


def points_to_grid(X, Y, grid_shape, background=torch.tensor([0.0, 0.0, 0.0])):
    """Converts points to a grid (undo mask select from datasplit)"""

    batch_size, _, y_dim = Y.shape
    X = X.clone()
    background = background.view(1, *(1 for _ in grid_shape), y_dim).repeat(
        batch_size, *grid_shape, 1
    )

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
