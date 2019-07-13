"""
Script to run the "small extended GNP" experiments on 1D dataset.
"""
from argparse import ArgumentParser

import os
from os.path import dirname, abspath
base_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
os.chdir(base_dir)

import sys
sys.path.append("notebooks")
sys.path.append(".")

import torch

N_THREADS = 8
torch.set_num_threads(N_THREADS)

from neuralproc import GlobalNeuralProcess
from neuralproc.utils.helpers import change_param
from neuralproc.utils.datasplit import CntxtTrgtGetter, GetRandomIndcs
from neuralproc.predefined import UnetCNN

from ntbks_helpers import get_gp_datasets, get_gp_datasets_varying, train_all_models_


parser = ArgumentParser()
parser.add_argument("-k", "--run", help="Run number", default=0, type=int)
args = parser.parse_args()

### Datasets ###
X_DIM = 1  # 1D spatial input
Y_DIM = 1  # 1D regression
N_POINTS = 128
N_SAMPLES = 100000  # this is a lot and can work with less
datasets = get_gp_datasets(n_samples=N_SAMPLES, n_points=N_POINTS)
datasets.update(get_gp_datasets_varying(n_samples=N_SAMPLES, n_points=N_POINTS))

contexts_getter = GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=.5)
targets_getter = GetRandomIndcs(min_n_indcs=0.5, max_n_indcs=0.99)
get_cntxt_trgt = CntxtTrgtGetter(contexts_getter=contexts_getter,
                                 targets_getter=targets_getter,
                                 is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

### Models ###
gnp_kwargs = dict(r_dim=16,
                  get_cntxt_trgt=get_cntxt_trgt,
                  TmpSelfAttn=change_param(UnetCNN,
                                           Conv=torch.nn.Conv1d,
                                           Pool=torch.nn.MaxPool1d,
                                           upsample_mode="linear",
                                           n_layers=14,
                                           is_double_conv=True,
                                           bottleneck=None,
                                           is_depth_separable=True,
                                           Normalization=torch.nn.BatchNorm1d,
                                           is_chan_last=True,
                                           kernel_size=7))

# initialize one model for each dataset
data_models = {name: (GlobalNeuralProcess(X_DIM, Y_DIM, **gnp_kwargs), data)
               for name, data in datasets.items()}

### Training ###
info = train_all_models_(data_models,
                         "results/attn_conv_compare/data_1D/run_k{}/extended_gnp_large".format(args.run),
                         is_retrain=True,
                         is_progress_bar=False)  # if false load precomputed
