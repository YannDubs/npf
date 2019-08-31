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
                  Conv=torch.nn.Conv2d,
                  Normalization=torch.nn.BatchNorm2d,  # ??
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


def get_models(model_names):
    models = dict()
    models_kwargs = dict()

    if "AttnCNP" in model_names:
        models["AttnCNP"] = partial(AttentiveNeuralProcess,
                                    x_dim=X_DIM,
                                    attention="transformer",
                                    **MODELS_KWARGS)
        models_kwargs["AttnCNP"] = dict(batch_size=32)

    if "SelfAttnCNP" in model_names:
        AttnCNP = get_models(["AttnCNP"])[0]["AttnCNP"]
        models["SelfAttnCNP"] = partial(AttnCNP,
                                        XYEncoder=merge_flat_input(SelfAttention,
                                                                   is_sum_merge=True))
        # use smaller batch size because memory ++
        models_kwargs["SelfAttnCNP"] = dict(batch_size=8)

    # work directly with masks
    masked_collate = cntxt_trgt_collate(GET_CNTXT_TRGT, is_return_masks=True)

    if "GridedCCP" in model_names:
        models["GridedCCP"] = partial(RegularGridsConvolutionalProcess,
                                      x_dim=X_DIM,
                                      # depth separable resnet
                                      PseudoTransformer=partial(CNN, n_blocks=7, **CNN_KWARGS),
                                      **MODELS_KWARGS)

        models_kwargs["GridedCCP"] = dict(iterator_train__collate_fn=masked_collate,
                                          iterator_valid__collate_fn=masked_collate)

    PseudoTransformerUnetCNN = partial(UnetCNN,
                                       Pool=torch.nn.MaxPool2d,
                                       upsample_mode="bilinear",
                                       max_nchannels=64,  # use constant number of channels and chosen to have similar # param
                                       n_blocks=9,
                                       **CNN_KWARGS)

    if "GridedUnetCCP" in model_names:
        models["GridedUnetCCP"] = partial(RegularGridsConvolutionalProcess,
                                          x_dim=X_DIM,
                                          # Unet CNN with depth separable resnet blocks
                                          PseudoTransformer=PseudoTransformerUnetCNN,
                                          **MODELS_KWARGS)
        models_kwargs["GridedUnetCCP"] = dict(iterator_train__collate_fn=masked_collate,
                                              iterator_valid__collate_fn=masked_collate)

    if "GridedSharedUnetCCP" in model_names:
        models["GridedSharedUnetCCP"] = partial(RegularGridsConvolutionalProcess,
                                                x_dim=X_DIM,
                                                # Unet CNN with depth separable resnet blocks
                                                PseudoTransformer=partial(PseudoTransformerUnetCNN,
                                                                          is_force_same_bottleneck=True),
                                                **MODELS_KWARGS)

        # repreat the batch twice => every function has 2 different cntxt and trgt samples
        repeat_collate = cntxt_trgt_collate(GET_CNTXT_TRGT,
                                            is_return_masks=True,
                                            is_repeat_batch=True)
        models_kwargs["GridedSharedUnetCCP"] = dict(iterator_train__collate_fn=repeat_collate,
                                                    iterator_valid__collate_fn=repeat_collate,
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
                     batch_size=64,
                     seed=123,
                     **kwargs)


def main(args):
    print(args)
    train_datasets, test_datasets, datasets_kwargs = get_datasets(args.datasets)
    models, models_kwargs = get_models(args.models)
    callbacks = [ProgressBar()] if args.is_progressbar else []
    train(models, train_datasets,
          datasets_kwargs=datasets_kwargs,
          models_kwargs=models_kwargs,
          callbacks=callbacks)


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
    parser.add_argument('-r', '--runs',
                        default=1,
                        type=int,
                        help='Number of runs.')
    parser.add_argument('--is-progressbar',
                        action='store_true',
                        help='Whether to use a progressbar.')
    args = parser.parse_args(args_to_parse)
    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    main(args)
