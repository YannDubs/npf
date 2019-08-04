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
from neuralproc.utils.setcnn import SetConv, GaussianRBF
from neuralproc.predefined import CNN

from ntbks_helpers import get_gp_datasets, get_gp_datasets_varying, train_all_models_


parser = ArgumentParser()
parser.add_argument("-k", "--run", help="Run number", default=0, type=int)
args = parser.parse_args()

### Datasets ###
X_DIM = 1  # 1D spatial input
Y_DIM = 1  # 1D regression
N_POINTS = 128
N_SAMPLES = 100000  # this is a lot and can work with less
N_DIFF_HYP = 1000
datasets = get_gp_datasets_varying(n_samples=N_SAMPLES, n_points=N_POINTS,
                                   n_diff_kernel_hyp=N_DIFF_HYP, save_file='data/gp_dataset.hdf5')

contexts_getter = GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=.5)
targets_getter = GetRandomIndcs(min_n_indcs=0.5, max_n_indcs=0.99)
get_cntxt_trgt = CntxtTrgtGetter(contexts_getter=contexts_getter,
                                 targets_getter=targets_getter,
                                 is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

### Models ###
gnp_kwargs = dict(r_dim=64,
                  keys_to_tmp_attn=change_param(SetConv, is_vanilla=True,
                                                RadialBasisFunc=GaussianRBF),
                  TmpSelfAttn=change_param(CNN,
                                           Conv=torch.nn.Conv1d,
                                           n_layers=5,
                                           is_depth_separable=False,
                                           Normalization=torch.nn.Identity,
                                           is_chan_last=True,
                                           kernel_size=11),
                  tmp_to_queries_attn=change_param(SetConv, is_vanilla=True,
                                                   RadialBasisFunc=GaussianRBF),
                  is_skip_tmp=False,
                  is_use_x=False,
                  get_cntxt_trgt=get_cntxt_trgt,
                  is_encode_xy=False)

# initialize one model for each dataset
data_models = {name: (GlobalNeuralProcess(X_DIM, Y_DIM, **gnp_kwargs), data)
               for name, data in datasets.items()}

### Training ###
DIR = "results/attn_conv_compare_large/data_1D"
info = train_all_models_(data_models,
                         "{}/run_k{}/vanilla_gnp".format(DIR, args.run),
                         is_retrain=True,
                         is_progress_bar=False)  # if false load precomputed
