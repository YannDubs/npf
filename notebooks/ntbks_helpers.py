import sys
sys.path.append("..")

from functools import partial

from sklearn.gaussian_process.kernels import (RBF, Matern, ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)

from skorch import NeuralNet

from neuralproc import NeuralProcessLoss, NeuralProcess, merge_flat_input
from neuralproc.predefined import MLP
from neuralproc.utils.datasplit import CntxtTrgtGetter, GetRandomIndcs, get_all_indcs

from utils.data import GPDataset


### TUTORIAL 1 ###
# DATA
def get_gp_datasets(n_samples=100000, n_points=128, save_file=None, **kwargs):
    """
    Return different 1D functions sampled from GPs with the following kernels:
    "rbf", "periodic", "non-stationary", "matern", "noisy-matern".

    Note
    ----
    - Hyper parameters chosen such that highest frequency is around 50.
    Tested empirically.
    """
    datasets = dict()
    kwargs.update(dict(n_samples=n_samples,
                       n_points=n_points,
                       is_vary_kernel_hyp=False))

    def add_dataset_(name, kernel, save_file=save_file):
        if save_file is not None:
            save_file = (save_file, name)
        datasets[name] = GPDataset(kernel=kernel, save_file=save_file, **kwargs)

    add_dataset_("Fixed_RBF_Kernel", RBF(length_scale=.1))
    """
    add_dataset_("Fixed_Periodic_Kernel",
                 ExpSineSquared(length_scale=.3, periodicity=1.0))
    add_dataset_("Fixed_Matern_Kernel",
                 Matern(length_scale=.1, nu=1.5))
    add_dataset_("Fixed_Noisy_Matern_Kernel",
                 (WhiteKernel(noise_level=.1) +
                  Matern(length_scale=.1, nu=1.5)))
    """
    return datasets


def get_gp_datasets_varying(n_samples=10000, n_points=128, save_file=None, **kwargs):
    """
    Return different 1D functions sampled from GPs with the following kernels:
    "rbf", "periodic", "non-stationary", "matern", "noisy-matern" with varying
    hyperparameters.
    """
    datasets = dict()
    kwargs.update(dict(n_samples=n_samples,
                       n_points=n_points,
                       is_vary_kernel_hyp=True))

    def add_dataset_(name, kernel, save_file=save_file):
        if save_file is not None:
            save_file = (save_file, name)
        datasets[name] = GPDataset(kernel=kernel, save_file=save_file, **kwargs)

    add_dataset_("RBF_Kernel", RBF(length_scale_bounds=(.02, .3)))
    add_dataset_("Periodic_Kernel",
                 ExpSineSquared(length_scale_bounds=(.2, .5), periodicity_bounds=(.5, 2.)))
    add_dataset_("Matern_Kernel",
                 Matern(length_scale_bounds=(.03, .3), nu=1.5))
    add_dataset_("Noisy_Matern_Kernel",
                 (WhiteKernel(noise_level_bounds=(.05, .7)) +
                  Matern(length_scale_bounds=(.03, .3), nu=1.5)))

    return datasets


# MODEL
get_cntxt_trgt = CntxtTrgtGetter(contexts_getter=GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=0.5),
                                 targets_getter=get_all_indcs,
                                 is_add_cntxts_to_trgts=False)

CNP_KWARGS = dict(XEncoder=partial(MLP, n_hidden_layers=1),
                  XYEncoder=merge_flat_input(partial(MLP, n_hidden_layers=2,
                                                     is_force_hid_smaller=True),
                                             is_sum_merge=True),
                  Decoder=merge_flat_input(partial(MLP, n_hidden_layers=2,
                                                   is_force_hid_smaller=True),
                                           is_sum_merge=True),
                  r_dim=128,
                  encoded_path="deterministic")
