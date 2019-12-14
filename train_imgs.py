import os
from os.path import dirname, abspath
import sys
import copy


from functools import partial
import argparse

import torch
import skorch
from skorch.callbacks import ProgressBar
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelSpreading
import numpy as np

from neuralproc import RegularGridsConvolutionalProcess, AttentiveNeuralProcess, NeuralProcessLoss
from neuralproc.predefined import UnetCNN, CNN, SelfAttention, MLP, ResConvBlock, GaussianConv2d
from neuralproc import merge_flat_input
from neuralproc.utils.datasplit import GridCntxtTrgtGetter, RandomMasker, no_masker, half_masker
from neuralproc.utils.predict import GenAllAutoregressivePixel
from neuralproc.training import NeuralNetTransformer
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
from utils.data.helpers import train_dev_split, make_ssl_targets, DatasetHelper
from utils.data.dataloader import cntxt_trgt_collate

SCORE_FILENAME = "score.csv"

def _update_dict(d, update):
    """Update a dictionary not in place."""
    d = d.copy()
    d.update(update)
    return d


def add_y_dim(models, datasets):
    """Add y_dim to all of the models depending on the dataset."""
    first_el = lambda t : t [0]if isinstance(t,tuple) else t
    return {
        data_name: {
            model_name: partial(model, y_dim=first_el(data_train).shape[0])
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
    # repeat the batch twice => every function has 2 different cntxt and trgt samples
    repeat_collate = cntxt_trgt_collate(
        get_cntxt_trgt, is_return_masks=True, is_duplicate_batch=True
    )

    if "AttnCNP" in model_name:
        dflt_kwargs = dict(
            iterator_train__collate_fn=dflt_collate, iterator_valid__collate_fn=dflt_collate
        )

    elif model_name in ["GridedCCP", "GridedCCPUnet"]:
        dflt_kwargs = dict(
            iterator_train__collate_fn=masked_collate, iterator_valid__collate_fn=masked_collate
        )

    elif model_name == "GridedCCPUnetShared":
        dflt_kwargs = dict(
            iterator_train__collate_fn=repeat_collate,
            iterator_valid__collate_fn=masked_collate,
            batch_size=kwargs.get("batch_size", 16) // 2,
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
    r_dim=128,
    pre_r_dim=32,
):
    """Return the correct model."""

    # PARAMETERS
    x_dim = 2
    neuralproc_kwargs = dict(
        r_dim=r_dim,
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

    elif "GridedCCP" in model_name:
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

        cnn_kwargs = dict(
            ConvBlock=ResConvBlock,
            Conv=make_padded_conv(nn.Conv2d, Padder),
            is_chan_last=True,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            is_bias=is_bias,
        )

        unetcnn_kwargs = dict(
            Pool=torch.nn.MaxPool2d, upsample_mode="bilinear", max_nchannels=r_dim, is_return_rep=True
        )  # use constant number of channels and chosen to have similar num param

        if model_name == "GridedCCP":
            PseudoTransformer = partial(CNN, **cnn_kwargs)

        elif model_name == "GridedCCPUnet":
            PseudoTransformer = partial(UnetCNN, **unetcnn_kwargs, **cnn_kwargs)
            neuralproc_kwargs["r_dim"] = pre_r_dim

        elif model_name == "GridedCCPUnetShared":
            PseudoTransformer = partial(
                UnetCNN, is_force_same_bottleneck=True, **unetcnn_kwargs, **cnn_kwargs
            )
            neuralproc_kwargs["r_dim"] = pre_r_dim

        model = partial(
            RegularGridsConvolutionalProcess,
            x_dim=x_dim,
            Conv=SetConv,
            PseudoTransformer=PseudoTransformer,
            is_density=not is_no_density,
            is_normalization=not is_no_normalization,
            **neuralproc_kwargs,
        )

    return model

def get_train_valid_test_dataset(dataset, valid_size=0.1, seed=123, **kwargs):
    """Return the correct instantiated train, validation, test dataset
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    
    valid_size : float or int, optional
        Size of the validation set. If float, should be between 0.0 and 1.0 and represent the 
        proportion of the dataset. If int, represents the absolute number of valid samples.
        
    seed : int, optional
        Random seed

    Returns
    -------
    datasets : dictionary of torch.utils.data.Dataset
        Dictionary of the `"train"`, `"valid"`, and `"valid"`.
    """
    datasets = dict()
    datasets["train"], datasets["valid"] = train_dev_split(
        get_dataset(dataset)(split="train", **kwargs), dev_size=valid_size, seed=seed
    )

    datasets["test"] = get_dataset(dataset)(split="test", **kwargs)

    return datasets



def train(models, train_datasets, **kwargs):
    """Train the model."""
    return train_models(
        train_datasets,
        add_y_dim(models, train_datasets),
        NeuralProcessLoss,
        train_split=skorch.dataset.CVSplit(0.1),  # use 10% of data for validation
        seed=123,
        **kwargs,
    )

def transform(transformer, datasets):
    """Transform dataset inplaces.
    
    Parameters
    ----------
    transformer : sklearn.base.TransformerMixin
        Transformer which transforms an input into a useful representation.

    datasets : dict of datasets.
        Dictionary of datasets to transform. Keys should be `train`, `test`, `valid`. 
    """
    out = {}
    for k, dataset in datasets.items():
        if isinstance(transformer, skorch.NeuralNet):
            data = transformer.transform(dataset)
        else:
            # sklearn
            data = transformer.transform(dataset.data.view(len(dataset), -1))
        out[k] = DatasetHelper(data.astype(np.float32), dataset.targets)
    return out


def reduce_datasets(datasets, n_labels={}, n_examples={}, seed=123):
    """Reduces the amount of examples or labels in each dataset.
    
    Parameters
    ----------
    datasets : dictionary of torch.utils.data.Dataset

    n_labels : dictionary of int, optional
        Number of labels to keep (making semi supervised) for every dataset. `-1` or missing keys 
        corresponds to no filtering. If there is a `kwargs` key, it will be given as kwargs. 

    n_examples : dictionary of int, optional
        Number of examples to keep for every dataset. `-1` or missing keys corresponds to no filtering.
        If there is a `kwargs` key, it will be given as kwargs. 

    seed : int, optional
        Random seed.

    Returns
    -------
    datasets : dictionary of torch.utils.data.Dataset
    """

    out = dict()
    for k, dataset in datasets.items():
        n_lab = n_labels.get(k, -1)
        if n_lab != -1:
            dataset = copy.deepcopy(dataset)
            dataset.targets = make_ssl_targets(
                dataset.targets, n_lab, seed=seed, **n_labels.get("kwargs", {})
            )

        n_ex = n_examples.get(k, -1)
        if n_ex != -1:
            _, dataset = train_dev_split(
                dataset, dev_size=n_ex, seed=seed, **n_labels.get("kwargs", {})
            )

        out[k] = dataset
    return out


def main(args):
    seed = 123+args.starting_run
    # DATA
    datasets = get_train_valid_test_dataset(args.dataset, is_augment=args.is_augment, seed=seed)
    train_dataset, valid_dataset, test_dataset = datasets["train"], datasets["valid"], datasets["test"]

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
        r_dim=args.r_dim,
    )

    model_kwargs = _get_train_kwargs(args.model, **dict(lr=args.lr, batch_size=args.batch_size))

    # TRAINING
    trainers = train(
        {args.name: model},
        {args.dataset: (train_dataset, valid_dataset)},
        test_datasets={args.dataset: test_dataset},
        models_kwargs={args.name: model_kwargs},
        callbacks=[ProgressBar()] if args.is_progressbar else [],
        runs=args.runs,
        starting_run=args.starting_run,
        max_epochs=args.max_epochs,
        is_continue_train=args.is_continue_train,
        patience=args.patience,
        chckpnt_dirname=args.chckpnt_dirname,
        is_retrain=not args.no_retrain,
        Trainer=NeuralNetTransformer
    )
    
    for k,trainer in trainers.items():
        if args.chckpnt_dirname is not None:
            with open(os.path.join(args.chckpnt_dirname + k, args.clf + "_" + SCORE_FILENAME), "w") as f:
                f.write("n_label,score")

        #datasets = reduce_datasets(datasets, n_examples=dict(train=500), seed=seed)
        if torch.cuda.is_available():
            trainer.module_.cuda()
        transformed_dataset = transform(trainer, datasets)
        print("transformed data")

        for n_label in [10,100,1000,-1]:
            if n_label != -1:
                n_label = n_label * datasets["train"].n_classes

            if args.clf == "mlp":
                clf_datasets = reduce_datasets(transformed_dataset, n_examples=dict(train=n_label), seed=seed)
            elif args.clf == "labelspread":
                breakpoint()
                clf_datasets = reduce_datasets(transformed_dataset, n_labels=dict(train=n_label), seed=seed)

            if args.clf == "mlp":
                clf = MLPClassifier(solver="lbfgs" if n_label < 1000 else "adam")
            elif args.clf == "labelspread":
                clf = LabelSpreading(kernel="knn", n_neighbors=1000, gamma=5, 
                                    n_jobs=-1, max_iter=30, alpha=0.2, tol=0.001)
            clf.fit(clf_datasets["train"].data, clf_datasets["train"].targets)
            score = clf.score(clf_datasets["test"].data, clf_datasets["test"].targets)

            if args.chckpnt_dirname is not None:
                with open(os.path.join(args.chckpnt_dirname + k, args.clf + "_" + SCORE_FILENAME), "a") as f:
                    f.write(f"{args.clf},{n_label},{score}")
                print(k, n_label, score)


def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model.",
        choices=["AttnCNP", "SelfAttnCNP", "GridedCCP", "GridedCCPUnet", "GridedCCPUnetShared"],
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset.",
        choices=["celeba32", "celeba64", "svhn", "mnist", "zs-multi-mnist", "celeba", "zs-mnist", "cifar10", "cifar100"],
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
        "--clf",
        default="mlp",
        type=str,
        help="Classifier.",
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
    general.add_argument(
        "--no-retrain",
        action="store_true",
        help="Whether not to retrain.",
    )
    general.add_argument(
        "--is-augment",
        action="store_true",
        help="Whether to augment the datset.",
    )
    general.add_argument(
        "--r-dim", default=128, type=int, help="Number of dimensions for representation."
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
