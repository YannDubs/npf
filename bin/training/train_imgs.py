import os
from os.path import dirname, abspath

base_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(base_dir)

import sys

sys.path.append("notebooks")
sys.path.append(".")


from functools import partial
import argparse

import torch
import skorch
from skorch.callbacks import ProgressBar
import torch.nn as nn
import torch.nn.functional as F

from neuralproc import RegularGridsConvolutionalProcess, AttentiveNeuralProcess, NeuralProcessLoss
from neuralproc.predefined import UnetCNN, CNN, SelfAttention, MLP, ResConvBlock
from neuralproc import merge_flat_input
from neuralproc.utils.datasplit import GridCntxtTrgtGetter, RandomMasker, no_masker, half_masker
from neuralproc.utils.helpers import (
    MultivariateNormalDiag,
    ProbabilityConverter,
    make_abs_conv,
    channels_to_2nd_dim,
    channels_to_last_dim,
)

from utils.train import train_models
from utils.data import get_dataset
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


def _get_dflt_train_kwargs(model_name, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    get_cntxt_trgt = GridCntxtTrgtGetter(
        context_masker=RandomMasker(min_nnz=0.01, max_nnz=0.5),
        target_masker=no_masker,
        is_add_cntxts_to_trgts=False,
    )

    dflt_collate = cntxt_trgt_collate(get_cntxt_trgt)
    masked_collate = cntxt_trgt_collate(get_cntxt_trgt, is_return_masks=True)
    # repeat the batch twice => every function has 2 different cntxt and trgt samples
    repeat_collate = cntxt_trgt_collate(get_cntxt_trgt, is_return_masks=True, is_repeat_batch=True)

    if model_name == "AttnCNP":
        dflt_kwargs = dict(
            batch_size=32,
            iterator_train__collate_fn=dflt_collate,
            iterator_valid__collate_fn=dflt_collate,
        )

    elif model_name == "SelfAttnCNP":
        dflt_kwargs = dict(
            batch_size=2,
            lr=5e-4,
            iterator_train__collate_fn=dflt_collate,
            iterator_valid__collate_fn=dflt_collate,
        )

    elif model_name in ["GridedCCP", "GridedUnetCCP"]:
        dflt_kwargs = dict(
            iterator_train__collate_fn=masked_collate, iterator_valid__collate_fn=masked_collate
        )

    elif model_name == "GridedSharedUnetCCP":
        dflt_kwargs = dict(
            iterator_train__collate_fn=repeat_collate,
            iterator_valid__collate_fn=masked_collate,
            batch_size=kwargs.get("batch_size", 64) // 2,
        )

    dflt_kwargs.update(kwargs)
    return dflt_kwargs


def get_model(
    model_name, min_sigma, n_blocks, init_kernel_size, kernel_size, img_shape, no_batchnorm
):

    # PARAMETERS
    x_dim = 2
    neuralproc_kwargs = dict(
        r_dim=128,
        # make sure output is in 0,1 as images preprocessed so
        pred_loc_transformer=lambda mu: torch.sigmoid(mu),
        pred_scale_transformer=lambda scale_trgt: min_sigma
        + (1 - min_sigma) * F.softplus(scale_trgt),
    )

    dflt_kernel_size = img_shape // 5
    if dflt_kernel_size % 2 == 0:
        dflt_kernel_size -= 1  # make sure odd

    if init_kernel_size is None:
        init_kernel_size = dflt_kernel_size
    if kernel_size is None:
        kernel_size = dflt_kernel_size

    SetConv = lambda y_dim: make_abs_conv(nn.Conv2d)(
        y_dim,
        y_dim,
        groups=y_dim,
        kernel_size=init_kernel_size,
        padding=init_kernel_size // 2,
        bias=False,
    )

    cnn_kwargs = dict(
        ConvBlock=ResConvBlock,
        Conv=nn.Conv2d,
        Normalization=nn.Identity if no_batchnorm else nn.BatchNorm2d,
        is_chan_last=True,
        kernel_size=kernel_size,
        n_blocks=n_blocks,
    )

    PseudoTransformerUnetCNN = partial(
        UnetCNN,
        Pool=torch.nn.MaxPool2d,
        upsample_mode="bilinear",
        max_nchannels=64,  # use constant number of channels and chosen to have similar num param
        **cnn_kwargs,
    )

    # MODEL
    if model_name == "AttnCNP":
        model = partial(
            AttentiveNeuralProcess, x_dim=x_dim, attention="transformer", **neuralproc_kwargs
        )

    elif model_name == "SelfAttnCNP":
        AttnCNP = get_model("AttnCNP")
        model = partial(AttnCNP, XYEncoder=merge_flat_input(SelfAttention, is_sum_merge=True))

    elif model_name == "GridedCCP":
        model = partial(
            RegularGridsConvolutionalProcess,
            x_dim=x_dim,
            Conv=SetConv,
            PseudoTransformer=partial(CNN, **cnn_kwargs),
            **neuralproc_kwargs,
        )

    elif model_name == "GridedUnetCCP":
        model = partial(
            RegularGridsConvolutionalProcess,
            x_dim=x_dim,
            PseudoTransformer=PseudoTransformerUnetCNN,
            **neuralproc_kwargs,
        )

    elif model_name == "GridedSharedUnetCCP":
        model = partial(
            RegularGridsConvolutionalProcess,
            x_dim=x_dim,
            # Unet CNN with depth separable resnet blocks
            PseudoTransformer=partial(PseudoTransformerUnetCNN, is_force_same_bottleneck=True),
            **neuralproc_kwargs,
        )

    return model


def get_train_test_dataset(dataset):
    """Return the correct instantiated train and test datasets."""
    try:
        train_dataset = get_dataset(dataset)(split="train")
        test_dataset = get_dataset(dataset)(split="test")
    except:
        train_dataset, test_dataset = train_dev_split(
            get_dataset(dataset)(), dev_size=0.1, is_stratify=False
        )

    return train_dataset, test_dataset


def train(models, train_datasets, **kwargs):
    _ = train_models(
        train_datasets,
        add_y_dim(models, train_datasets),
        NeuralProcessLoss,
        chckpnt_dirname="results/neural_process_imgs/",
        is_retrain=True,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of data for validation
        patience=10,
        seed=123,
        **kwargs,
    )


def main(args):

    # DATA
    train_datasets, test_datasets = get_train_test_dataset(args.datasets)

    model = get_model(
        args.model,
        min_sigma=args.min_sigma,
        n_blocks=args.n_blocks,
        init_kernel_size=args.init_kernel_size,
        kernel_size=args.kernel_size,
        batch_size=args.batch_size,
        lr=args.lr,
        no_batchnorm=args.no_batchnorm,
    )

    model_kwargs = _get_dflt_train_kwargs(
        args.model, **dict(lr=args.lr, batch_size=args.batch_size)
    )

    # TRAINING
    train(
        {args.name: model},
        train_datasets,
        test_datasets=test_datasets,
        models_kwargs={args.name: model_kwargs},
        callbacks=[ProgressBar()] if args.is_progressbar else [],
        runs=args.runs,
        starting_run=args.starting_run,
        max_epochs=args.max_epochs,
    )


def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Model.",
        choices=["AttnCNP", "SelfAttnCNP", "GridedCCP", "GridedUnetCCP", "GridedSharedUnetCCP"],
        required=True,
    )
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        type=str,
        help="Datasets.",
        choices=["celeba32", "celeba64", "svhn", "mnist", "zs-multi-mnist"],
        required=True,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name of the model for saving. By default parameter from `--model`.",
    )
    parser.add_argument("--lr", type=float, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, help="Batch size.")
    parser.add_argument("--max-epochs", default=100, type=int, help="Max number of epochs.")
    parser.add_argument("--runs", default=1, type=int, help="Number of runs.")
    parser.add_argument(
        "--starting-run",
        default=0,
        type=int,
        help="Starting run. This is useful if a couple of runs have already been trained, and you want to continue from there.",
    )
    parser.add_argument(
        "--min-sigma",
        default=0.1,
        type=float,
        help="Lowest bound on the std that the model can predict.",
    )
    parser.add_argument(
        "--is-progressbar", action="store_true", help="Whether to use a progressbar."
    )

    # General options
    ccp = parser.add_argument_group("CCP Options")
    ccp.add_argument("--n-blocks", default=5, type=int, help="Number of blocks to use in the CNN.")
    ccp.add_argument("--init-kernel-size", type=int, help="Kernel size to use for the set cnn.")
    ccp.add_argument("--kernel-size", type=int, help="Kernel size to use for the whole CNN.")
    ccp.add_argument(
        "--no-batchnorm", action="store_true", help="Whether to remove batchnorm when training CCP."
    )
    ccp.add_argument(
        "--is-receptivefield",
        action="store_true",
        help="Numerically estimates the receptive field of the model.",
    )

    args = parser.parse_args(args_to_parse)

    if "name" not in args:
        args.name = args.model

    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
