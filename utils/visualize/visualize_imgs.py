import random
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np
from skorch.dataset import unpack_data
import seaborn as sns
import matplotlib.ticker as ticker

from neuralproc.utils.helpers import prod, channels_to_2nd_dim
from neuralproc.utils.predict import VanillaPredictor
from neuralproc.utils.datasplit import GridCntxtTrgtGetter
from neuralproc import RegularGridsConvolutionalProcess
from utils.data import cntxt_trgt_collate
from utils.helpers import set_seed, tuple_cont_to_cont_tuple
from utils.train import EVAL_FILENAME

__all__ = ["plot_dataset_samples_imgs", "plot_posterior_img", "plot_qualitative_with_kde"]

DFLT_FIGSIZE = (17, 9)


def remove_axis(ax, is_rm_ticks=True, is_rm_spines=True):
    """Remove all axis but not the labels."""
    if is_rm_spines:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if is_rm_ticks:
        ax.tick_params(bottom="off", left="off")


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
    is_hrztl_cat=False,
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
        device = next(model.parameters()).device
        mask_cntxt = get_cntxt_trgt["X_cntxt"].to(device)
        X = get_cntxt_trgt["Y_cntxt"].to(device)
        mask_trgt = get_cntxt_trgt["X_trgt"].to(device)
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
        out_cntxt, _ = points_to_grid(
            mask_cntxt, X, data.shape[1:], background=data.missing_px_color
        )

        out_pred, _ = points_to_grid(
            mask_trgt, mean_y, data.shape[1:], background=data.missing_px_color
        )

    outs = [out_cntxt, out_pred]

    grid = make_grid(
        channels_to_2nd_dim(torch.cat(outs, dim=0)),
        nrow=n_plots * 2 if is_hrztl_cat else n_plots,
        pad_value=1.0,
    )

    if is_return:
        return grid

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis("off")


# TO CLEAN
def plot_qualitative_with_kde(
    named_trainer,
    dataset,
    named_trainer_compare=None,
    n_images=8,
    percentiles=None,  # if None uses uniform linspace from n_images
    figsize=DFLT_FIGSIZE,
    title=None,
    seed=123,
    height_ratios=[1, 3],
    font_size=14,
    h_pad=-3,
    x_lim={},
    is_smallest_xrange=False,
    kdeplot_kwargs={},
    **kwargs
):
    """
    Plot qualitative samples using `plot_posterior_img` but select the samples and mask to plot
    given the score at test time.
    
    VERY DIRTY
    """

    if percentiles is not None:
        n_images = len(percentiles)

    plt.rcParams.update({"font.size": font_size})
    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": height_ratios})

    def _plot_kde_loglike(name, trainer):
        chckpnt_dirname = dict(trainer.callbacks_)["Checkpoint"].dirname
        test_eval_file = os.path.join(chckpnt_dirname, EVAL_FILENAME)
        test_loglike = np.loadtxt(test_eval_file, delimiter=",")
        sns.kdeplot(test_loglike, ax=axes[0], shade=True, label=name, cut=0, **kdeplot_kwargs)
        sns.despine()
        return test_loglike

    def _grid_to_points(selected_data):
        for i in range(n_images):
            X = selected_data["Y_cntxt"][i]
            X_cntxt, Y_cntxt = GridCntxtTrgtGetter.select(
                None, X, None, selected_data["X_cntxt"][i]
            )
            X_trgt, Y_trgt = GridCntxtTrgtGetter.select(None, X, None, selected_data["X_trgt"][i])
            yield dict(X_cntxt=X_cntxt, Y_cntxt=Y_cntxt, X_trgt=X_trgt, Y_trgt=Y_trgt)

    def _plot_posterior_img_selected(name, trainer, selected_data, is_grided_trainer):
        is_uniform_grid = isinstance(trainer.module_, RegularGridsConvolutionalProcess)

        kwargs["img_indcs"] = []
        kwargs["is_uniform_grid"] = is_uniform_grid
        kwargs["is_return"] = True

        if not is_uniform_grid:
            if is_grided_trainer:
                grids = [
                    plot_posterior_img(dataset, data, trainer.module_.cpu(), **kwargs)
                    for i, data in enumerate(_grid_to_points(selected_data))
                ]
            else:
                grids = [
                    plot_posterior_img(
                        dataset,
                        {k: v[i] for k, v in selected_data.items()},
                        trainer.module_.cpu(),
                        **kwargs
                    )
                    for i in range(n_images)
                ]

            # images are padded by 2 pixels inbetween each but here you concatenate => will pad twice
            # => remove all the rleft padding for each besides first
            grids = [g[..., 2:] if i != 0 else g for i, g in enumerate(grids)]
            return torch.cat(grids, axis=-1)

        elif is_uniform_grid:
            if not is_grided_trainer:
                grids = []
                for i in range(n_images):

                    _, X_cntxt = points_to_grid(
                        selected_data["X_cntxt"][i],
                        selected_data["Y_cntxt"][i],
                        dataset.shape[1:],
                        background=torch.tensor([0.0] * dataset.shape[0]),
                    )
                    Y_trgt, X_trgt = points_to_grid(
                        selected_data["X_trgt"][i],
                        selected_data["Y_trgt"][i],
                        dataset.shape[1:],
                        background=torch.tensor([0.0] * dataset.shape[0]),
                    )

                    grids.append(
                        plot_posterior_img(
                            dataset,
                            dict(
                                X_cntxt=X_cntxt,
                                Y_cntxt=Y_trgt,  # Y_trgt is all X because no masking for target (assumption)
                                X_trgt=X_trgt,
                                Y_trgt=Y_trgt,
                            ),
                            trainer.module_.cpu(),
                            **kwargs
                        )
                    )

                grids = [g[..., 2:] if i != 0 else g for i, g in enumerate(grids)]

                return torch.cat(grids, axis=-1)
            else:
                return plot_posterior_img(
                    dataset,
                    {k: torch.cat(v, dim=0) for k, v in selected_data.items()},
                    trainer.module_.cpu(),
                    **kwargs
                )

    name, trainer = named_trainer
    test_loglike = _plot_kde_loglike(name, trainer)

    if named_trainer_compare is not None:
        left = axes[0].get_xlim()[0]
        _ = _plot_kde_loglike(*named_trainer_compare)
        axes[0].set_xlim(left=left)  # left bound by first model to not look strange

    if len(x_lim) != 0:
        axes[0].set_xlim(**x_lim)

    if percentiles is not None:
        idcs = []
        values = []
        for i, p in enumerate(percentiles):
            # value closest to percentile
            percentile_val = np.percentile(test_loglike, p, interpolation="nearest")
            idcs.append(np.argwhere(test_loglike == percentile_val).item())
            values.append(percentile_val)
        sorted_idcs = list(np.sort(idcs))[::-1]

        if is_smallest_xrange:
            axes[0].set_xlim(left=values[0] - 0.05, right=values[-1] + 0.05)
    else:
        # find indices such that same space between all
        values = np.linspace(test_loglike.min(), test_loglike.max(), n_images)
        idcs = [(np.abs(test_loglike - v)).argmin() for v in values]
        sorted_idcs = list(np.sort(idcs))[::-1]

    axes[0].set_ylabel("Density")
    axes[0].set_xlabel("Test Log-Likelihood")

    selected_data = []

    set_seed(seed)  # make sure same order and indices for cntxt and trgt
    i = -1

    saved_values = []
    queue = sorted_idcs.copy()
    next_idx = queue.pop()

    for data in trainer.get_iterator(dataset, training=False):
        Xi, yi = unpack_data(data)

        for cur_idx in range(yi.size(0)):
            i += 1
            if next_idx != i:
                continue

            selected_data.append({k: v[cur_idx : cur_idx + 1, ...] for k, v in Xi.items()})

            if len(queue) == 0:
                break
            else:
                next_idx = queue.pop()

    # puts back to non sorted array
    selected_data = [selected_data[sorted_idcs[::-1].index(idx)] for idx in idcs]

    selected_data = {k: v for k, v in tuple_cont_to_cont_tuple(selected_data).items()}

    for v in values:
        axes[0].axvline(v, linestyle=":", alpha=0.7, c="tab:green")

    axes[0].legend(loc="upper left")

    if title is not None:
        axes[0].set_title(title, fontsize=18)

    is_grided_trainer = isinstance(trainer.module_, RegularGridsConvolutionalProcess)
    grid = _plot_posterior_img_selected(name, trainer, selected_data, is_grided_trainer)

    middle_img = dataset.shape[1] // 2 + 1
    y_ticks = [middle_img, middle_img * 3]
    y_ticks_labels = ["Context", name]

    if named_trainer_compare is not None:
        grid_compare = _plot_posterior_img_selected(
            *named_trainer_compare, selected_data, is_grided_trainer
        )
        grid = torch.cat((grid, grid_compare[:, grid_compare.size(1) // 2 + 1 :, :]), dim=1)

        y_ticks += [middle_img * 5]
        y_ticks_labels += [named_trainer_compare[0]]

    axes[1].imshow(grid.permute(1, 2, 0).numpy())

    axes[1].yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    axes[1].set_yticklabels(y_ticks_labels, rotation="vertical", va="center")

    remove_axis(axes[1])

    if percentiles is not None:
        axes[1].xaxis.set_major_locator(
            ticker.FixedLocator(
                [(dataset.shape[2] // 2 + 1) * (i * 2 + 1) for i, p in enumerate(percentiles)]
            )
        )
        axes[1].set_xticklabels(["{}%".format(p) for p in percentiles])
    else:
        axes[1].set_xticks([])

    fig.tight_layout(h_pad=h_pad)
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

    X = X.round().long()
    idcs = idcs_grid_to_idcs_flatten(X, grid_shape)

    background = background.view(batch_size, -1, y_dim)
    mask = torch.zeros(batch_size, background.size(1), 1).bool()

    for b in range(batch_size):
        background[b, idcs[b], :] = Y[b]
        mask[b, idcs[b], :] = True

    background = background.view(batch_size, *grid_shape, y_dim)

    return background, mask.view(batch_size, *grid_shape, 1)

