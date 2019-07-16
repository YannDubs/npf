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

from neuralproc import AttentiveNeuralProcess
from neuralproc.utils.helpers import change_param
from neuralproc.utils.datasplit import CntxtTrgtGetter, GetRandomIndcs

from ntbks_helpers import (get_gp_datasets, get_gp_datasets_varying,
                           train_all_models_, CNP_KWARGS)


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
targets_getter = GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=0.5)
get_cntxt_trgt = CntxtTrgtGetter(contexts_getter=contexts_getter,
                                 targets_getter=targets_getter,
                                 is_add_cntxts_to_trgts=True)

### Models ###
anp_kwargs = dict(r_dim=32,
                  get_cntxt_trgt=get_cntxt_trgt,
                  encoded_path="deterministic",
                  attention="multihead",
                  is_relative_pos=False)

# initialize one model for each dataset
data_models = {name: (AttentiveNeuralProcess(X_DIM, Y_DIM, **anp_kwargs), data)
               for name, data in datasets.items()}

### Training ###
info = train_all_models_(data_models,
                         "results/attn_conv_compare/data_1D/run_k{}/vanilla_anp".format(args.run),
                         is_retrain=True,
                         is_progress_bar=False)  # if false load precomputed
