import os
from os.path import dirname, abspath
import sys


from functools import partial
import argparse

import torch
import skorch
from skorch.callbacks import ProgressBar
import torch.nn as nn
import torch.nn.functional as F

from neuralproc import RegularGridsConvolutionalProcess, AttentiveNeuralProcess, NeuralProcessLoss
from neuralproc.predefined import CNN, SelfAttention, MLP, ResConvBlock, GaussianConv2d
from neuralproc import merge_flat_input
from neuralproc.utils.datasplit import GridCntxtTrgtGetter, RandomMasker, no_masker, half_masker
from neuralproc.utils.predict import GenAllAutoregressivePixel
from neuralproc.utils.helpers import (
    MultivariateNormalDiag,
    ProbabilityConverter,
    make_abs_conv,
    make_padded_conv,
    channels_to_2nd_dim,
    channels_to_last_dim,
    CircularPad2d,
)

from utils.train import train_models
from utils.data import get_dataset, get_img_size
from utils.data.helpers import train_dev_split
from utils.data.dataloader import cntxt_trgt_collate


def _update_dict(d, update):
    """Update a dictionary not in place."""
    d = d.copy()
    d.update(update)
    return d


def add_y_dim(models, datasets):
    """Add y_dim to all of the models depending on the dataset."""
    return {
        data_name: {
            model_name: partial(model, y_dim=data_train.shape[0])
            for model_name, model in models.items()
        }
        for data_name, data_train in datasets.items()
    }


def _get_train_kwargs(model_name, **kwargs):
    """Return the model specific kwargs."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    get_cntxt_trgt = GridCntxtTrgtGetter(
        context_masker=RandomMasker(min_nnz=0.01, max_nnz=0.5),
        target_masker=no_masker,
        is_add_cntxts_to_trgts=False,
    )

    dflt_collate = cntxt_trgt_collate(get_cntxt_trgt)
    masked_collate = cntxt_trgt_collate(get_cntxt_trgt, is_return_masks=True)

    if "AttnCNP" in model_name:
        dflt_kwargs = dict(
            iterator_train__collate_fn=dflt_collate, iterator_valid__collate_fn=dflt_collate
        )

    elif "GridedCCP" in model_name:
        dflt_kwargs = dict(
            iterator_train__collate_fn=masked_collate, iterator_valid__collate_fn=masked_collate
        )

    dflt_kwargs.update(kwargs)
    return dflt_kwargs


def get_model(
    model_name,
    min_sigma=0.1,
    n_blocks=5,
    init_kernel_size=None,
    kernel_size=None,
    img_shape=(32, 32),
    is_no_normalization=False,
    is_no_density=False,
    is_rbf=False,
    is_no_abs=False,
    is_circular_padding=False,
    is_bias=True,
):
    """Return the correct model."""

    # PARAMETERS
    x_dim = 2
    neuralproc_kwargs = dict(
        r_dim=128,
        # make sure output is in 0,1 as images preprocessed so
        pred_loc_transformer=lambda mu: torch.sigmoid(mu),
        pred_scale_transformer=lambda scale_trgt: min_sigma
        + (1 - min_sigma) * F.softplus(scale_trgt),
    )

    # MODEL
    AttnCNP = partial(
        AttentiveNeuralProcess, x_dim=x_dim, attention="transformer", **neuralproc_kwargs
    )

    Padder = CircularPad2d if is_circular_padding else None

    if model_name == "AttnCNP":
        model = AttnCNP

    elif model_name == "SelfAttnCNP":
        model = partial(AttnCNP, XYEncoder=merge_flat_input(SelfAttention, is_sum_merge=True))

    elif model_name == "GridedCCP":
        denom = 10
        dflt_kernel_size = img_shape[-1] // denom  # currently assumes square images
        if dflt_kernel_size % 2 == 0:
            dflt_kernel_size -= 1  # make sure odd

        if init_kernel_size is None:
            init_kernel_size = dflt_kernel_size + 4
        if kernel_size is None:
            kernel_size = dflt_kernel_size

        if is_rbf:

            SetConv = lambda *args: make_padded_conv(GaussianConv2d, Padder)(
                kernel_size=init_kernel_size, padding=init_kernel_size // 2
            )
        elif is_no_abs:
            SetConv = lambda y_dim: make_padded_conv(nn.Conv2d, Padder)(
                y_dim,
                y_dim,
                groups=y_dim,
                kernel_size=init_kernel_size,
                padding=init_kernel_size // 2,
                bias=False,
            )

        else:
            SetConv = lambda y_dim: make_padded_conv(make_abs_conv(nn.Conv2d), Padder)(
                y_dim,
                y_dim,
                groups=y_dim,
                kernel_size=init_kernel_size,
                padding=init_kernel_size // 2,
                bias=False,
            )

        model = partial(
            RegularGridsConvolutionalProcess,
            x_dim=x_dim,
            Conv=SetConv,
            PseudoTransformer=partial(
                CNN,
                ConvBlock=ResConvBlock,
                Conv=make_padded_conv(nn.Conv2d, Padder),
                is_chan_last=True,
                kernel_size=kernel_size,
                n_blocks=n_blocks,
                is_bias=is_bias,
            ),
            is_density=not is_no_density,
            is_normalization=not is_no_normalization,
            **neuralproc_kwargs,
        )

    return model


def get_train_test_dataset(dataset):
    """Return the correct instantiated train and test datasets."""
    try:
        train_dataset = get_dataset(dataset)(split="train")
        test_dataset = get_dataset(dataset)(split="test")
    except TypeError:
        train_dataset, test_dataset = train_dev_split(
            get_dataset(dataset)(), dev_size=0.1, is_stratify=False
        )

    return train_dataset, test_dataset


def train(models, train_datasets, **kwargs):
    """Train the model."""
    _ = train_models(
        train_datasets,
        add_y_dim(models, train_datasets),
        NeuralProcessLoss,
        is_retrain=True,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of data for validation
        seed=123,
        **kwargs,
    )


def main(args):

    # DATA
    train_dataset, test_dataset = get_train_test_dataset(args.dataset)

    model = get_model(
        args.model,
        min_sigma=args.min_sigma,
        n_blocks=args.n_blocks,
        init_kernel_size=args.init_kernel_size,
        kernel_size=args.kernel_size,
        img_shape=get_img_size(args.dataset),
        is_no_density=args.is_no_density,
        is_no_normalization=args.is_no_normalization,
        is_rbf=args.is_rbf,
        is_no_abs=args.is_no_abs,
        is_circular_padding=args.is_circular_padding,
        is_bias=not args.is_no_bias,
    )

    model_kwargs = _get_train_kwargs(args.model, **dict(lr=args.lr, batch_size=args.batch_size))

    # TRAINING
    train(
        {args.name: model},
        {args.dataset: train_dataset},
        test_datasets={args.dataset: test_dataset},
        models_kwargs={args.name: model_kwargs},
        callbacks=[ProgressBar()] if args.is_progressbar else [],
        runs=args.runs,
        starting_run=args.starting_run,
        max_epochs=args.max_epochs,
        is_continue_train=args.is_continue_train,
        patience=args.patience,
        chckpnt_dirname=args.chckpnt_dirname,
    )


def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="Model.", choices=["AttnCNP", "SelfAttnCNP", "GridedCCP"]
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset.",
        choices=["celeba32", "celeba64", "svhn", "mnist", "zs-multi-mnist", "celeba", "zs-mnist"],
    )

    # General optional args
    general = parser.add_argument_group("General Options")
    general.add_argument(
        "--name",
        type=str,
        help="Name of the model for saving. By default parameter from `--model`.",
    )
    general.add_argument("--lr", type=float, help="Learning rate.")
    general.add_argument("--batch-size", type=int, help="Batch size.")
    general.add_argument("--max-epochs", default=100, type=int, help="Max number of epochs.")
    general.add_argument("--runs", default=1, type=int, help="Number of runs.")
    general.add_argument(
        "--starting-run",
        default=0,
        type=int,
        help="Starting run. This is useful if a couple of runs have already been trained, and you want to continue from there.",
    )
    general.add_argument(
        "--min-sigma",
        default=0.1,
        type=float,
        help="Lowest bound on the std that the model can predict.",
    )
    general.add_argument(
        "--chckpnt-dirname",
        default="results/iclr_imgs/",
        type=str,
        help="Checkpoint and result directory.",
    )
    general.add_argument("--patience", default=10, type=int, help="Patience for early stopping.")
    general.add_argument(
        "--is-progressbar", action="store_true", help="Whether to use a progressbar."
    )
    general.add_argument(
        "--is-continue-train",
        action="store_true",
        help="Whether to continue training from the last checkpoint of the previous run.",
    )

    # CCP options
    ccp = parser.add_argument_group("CCP Options")
    ccp.add_argument("--n-blocks", default=5, type=int, help="Number of blocks to use in the CNN.")
    ccp.add_argument("--init-kernel-size", type=int, help="Kernel size to use for the set cnn.")
    ccp.add_argument("--kernel-size", type=int, help="Kernel size to use for the whole CNN.")
    ccp.add_argument(
        "--is-rbf", action="store_true", help="Whether to use gaussian rbf as first layer."
    )
    ccp.add_argument(
        "--is-no-density", action="store_true", help="Whether not to add the density channel."
    )
    ccp.add_argument(
        "--is-no-normalization", action="store_true", help="Whether not to normalize."
    )
    ccp.add_argument(
        "--is-no-abs",
        action="store_true",
        help="Whether not to use absolute weights for the first layer.",
    )
    ccp.add_argument("--is-no-bias", action="store_true", help="Whether not to use bias.")
    ccp.add_argument(
        "--is-circular-padding", action="store_true", help="Whether to use reflect padding."
    )

    args = parser.parse_args(args_to_parse)

    if args.name is None:
        args.name = args.model

    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
