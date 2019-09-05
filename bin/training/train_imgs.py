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
from neuralproc.utils.helpers import (MultivariateNormalDiag, ProbabilityConverter,
                                      make_abs_conv, channels_to_2nd_dim,
                                      channels_to_last_dim)

from utils.train import train_models
from utils.data import get_dataset
from utils.data.helpers import train_dev_split
from utils.data.dataloader import cntxt_trgt_collate


X_DIM = 2
MODELS_KWARGS = dict(r_dim=128,
                     # make sure output is in 0,1 as images preprocessed so
                     pred_loc_transformer=lambda mu: torch.sigmoid(mu))
CNN_KWARGS = dict(ConvBlock=ResConvBlock,
                  Conv=nn.Conv2d,
                  Normalization=nn.BatchNorm2d,  # ??
                  is_chan_last=True,
                  kernel_size=11)
GET_CNTXT_TRGT = GridCntxtTrgtGetter(context_masker=RandomMasker(min_nnz=0.01, max_nnz=0.5),
                                     target_masker=no_masker,
                                     is_add_cntxts_to_trgts=False)


def add_y_dim(models, datasets):
    """Add y _dim to all ofthe models depending on the dataset."""
    return {data_name: {model_name: partial(model, y_dim=data_train.shape[0])
                        for model_name, model in models.items()}
            for data_name, data_train in datasets.items()}


def get_models(model_names, **kwargs):
    models = dict()
    models_kwargs = dict()

    if "AttnCNP" in model_names:
        models["AttnCNP"] = partial(AttentiveNeuralProcess,
                                    x_dim=X_DIM,
                                    attention="transformer",
                                    **MODELS_KWARGS,
                                    **kwargs)
        models_kwargs["AttnCNP"] = dict(batch_size=32)

    if "SelfAttnCNP" in model_names:
        AttnCNP = get_models(["AttnCNP"])[0]["AttnCNP"]
        models["SelfAttnCNP"] = partial(AttnCNP,
                                        XYEncoder=merge_flat_input(SelfAttention,
                                                                   is_sum_merge=True))
        # use smaller batch size because memory ++
        models_kwargs["SelfAttnCNP"] = dict(batch_size=16, lr=5e-4)

    # work directly with masks
    masked_collate = cntxt_trgt_collate(GET_CNTXT_TRGT, is_return_masks=True)

    if "GridedCCP" in model_names:
        models["GridedCCP"] = partial(RegularGridsConvolutionalProcess,
                                      x_dim=X_DIM,
                                      # depth separable resnet
                                      PseudoTransformer=partial(CNN, n_blocks=7, **CNN_KWARGS),
                                      **MODELS_KWARGS,
                                      **kwargs)

        models_kwargs["GridedCCP"] = dict(iterator_train__collate_fn=masked_collate,
                                          iterator_valid__collate_fn=masked_collate)

    PseudoTransformerUnetCNN = partial(UnetCNN,
                                       Pool=torch.nn.MaxPool2d,
                                       upsample_mode="bilinear",
                                       max_nchannels=64,  # use constant number of channels and chosen to have similar # param
                                       n_blocks=9,
                                       **CNN_KWARGS,
                                       **kwargs)

    if "GridedUnetCCP" in model_names:
        models["GridedUnetCCP"] = partial(RegularGridsConvolutionalProcess,
                                          x_dim=X_DIM,
                                          # Unet CNN with depth separable resnet blocks
                                          PseudoTransformer=PseudoTransformerUnetCNN,
                                          **MODELS_KWARGS,
                                          **kwargs)
        models_kwargs["GridedUnetCCP"] = dict(iterator_train__collate_fn=masked_collate,
                                              iterator_valid__collate_fn=masked_collate)

    if "GridedSharedUnetCCP" in model_names:
        models["GridedSharedUnetCCP"] = partial(RegularGridsConvolutionalProcess,
                                                x_dim=X_DIM,
                                                # Unet CNN with depth separable resnet blocks
                                                PseudoTransformer=partial(PseudoTransformerUnetCNN,
                                                                          is_force_same_bottleneck=True),
                                                **MODELS_KWARGS,
                                                **kwargs)

        # repreat the batch twice => every function has 2 different cntxt and trgt samples
        repeat_collate = cntxt_trgt_collate(GET_CNTXT_TRGT,
                                            is_return_masks=True,
                                            is_repeat_batch=True)
        models_kwargs["GridedSharedUnetCCP"] = dict(iterator_train__collate_fn=repeat_collate,
                                                    # don't repeat when eval
                                                    iterator_valid__collate_fn=masked_collate,
                                                    # like that actually same batch size
                                                    batch_size=32)

    return models, models_kwargs


def get_datasets(datasets):
    train_datasets = dict()
    test_datasets = dict()
    datasets_kwargs = dict()

    if "celeba32" in datasets:
        celeba32_train, celeba32_test = train_dev_split(get_dataset("celeba32")(),
                                                        dev_size=0.1, is_stratify=False)
        train_datasets["celeba32"] = celeba32_train
        test_datasets["celeba32"] = celeba32_test

    if "celeba64" in datasets:
        celeba64_train, celeba64_test = train_dev_split(get_dataset("celeba64")(),
                                                        dev_size=0.1, is_stratify=False)
        train_datasets["celeba64"] = celeba64_train
        test_datasets["celeba64"] = celeba64_test

    if "svhn" in datasets:
        train_datasets["svhn"] = get_dataset("svhn")(split="train")
        test_datasets["svhn"] = get_dataset("svhn")(split="test")

    if "mnist" in datasets:
        train_datasets["mnist"] = get_dataset("mnist")(split="train")
        test_datasets["mnist"] = get_dataset("mnist")(split="test")

    return train_datasets, test_datasets, datasets_kwargs


def train(models, train_datasets, **kwargs):
    _ = train_models(train_datasets,
                     add_y_dim(models, train_datasets),
                     NeuralProcessLoss,
                     chckpnt_dirname="results/neural_process_imgs/",
                     is_retrain=True,
                     train_split=skorch.dataset.CVSplit(0.1),  # use 10% of data for validation
                     iterator_train__collate_fn=cntxt_trgt_collate(GET_CNTXT_TRGT),
                     iterator_valid__collate_fn=cntxt_trgt_collate(GET_CNTXT_TRGT),
                     patience=10,
                     seed=123,
                     **kwargs)


def main(args):

    # DATA
    train_datasets, test_datasets, datasets_kwargs = get_datasets(args.datasets)

    # MODELS
    if args.no_batchnorm:
        CNN_KWARGS["Normalization"] = nn.Identity
    pred_scale_transformer = lambda scale_trgt: args.min_sigma + (1 - args.min_sigma
                                                                  ) * F.softplus(scale_trgt)
    models, models_kwargs = get_models(args.models,
                                       pred_scale_transformer=pred_scale_transformer)

    # TRAINING
    callbacks = [ProgressBar()] if args.is_progressbar else []
    train(models, train_datasets,
          datasets_kwargs=datasets_kwargs,
          models_kwargs=models_kwargs,
          callbacks=callbacks,
          runs=args.runs,
          starting_run=args.starting_run,
          max_epochs=args.max_epochs,
          lr=args.lr,
          batch_size=args.batch_size)


def parse_arguments(args_to_parse):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datasets',
                        nargs='+',
                        type=str,
                        help='Datasets.',
                        choices=['celeba32', 'celeba64', 'svhn', 'mnist'],
                        required=True)
    parser.add_argument('-m', '--models',
                        nargs='+',
                        type=str,
                        help='Models.',
                        choices=['AttnCNP', 'SelfAttnCNP', 'GridedCCP',
                                 'GridedUnetCCP', 'GridedSharedUnetCCP'],
                        required=True)
    parser.add_argument('-l', "--lr",
                        default=1e-3,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('-e', "--max-epochs",
                        default=100,
                        type=int,
                        help='Max number of epochs.')
    parser.add_argument('-b', "--batch-size",
                        default=64,
                        type=int,
                        help='Batch size.')
    parser.add_argument('-r', '--runs',
                        default=1,
                        type=int,
                        help='Number of runs.')
    parser.add_argument('--starting-run',
                        default=0,
                        type=int,
                        help='Starting run. This is useful if a couple of runs have already been trained, and you want to continue from there.')
    parser.add_argument('--min-sigma',
                        default=0.1,
                        type=float,
                        help='Lowest bound on the std that the model can predict.')
    parser.add_argument('--no-batchnorm',
                        action='store_true',
                        help='Whether to remove batchnorm when training CCP.')
    parser.add_argument('--is-progressbar',
                        action='store_true',
                        help='Whether to use a progressbar.')
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
