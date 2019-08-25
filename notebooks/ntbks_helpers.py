import sys
sys.path.append("..")

from sklearn.gaussian_process.kernels import (RBF, Matern, ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
import torch
from torch.distributions import Normal
from torch.optim import Adam
from skorch.callbacks import ProgressBar, Checkpoint
from skorch import NeuralNet

from neuralproc import NeuralProcessLoss, NeuralProcess, merge_flat_input
from neuralproc.predefined import MLP
from neuralproc.utils.datasplit import CntxtTrgtGetter, GetRandomIndcs
from neuralproc.utils.helpers import change_param

from utils.datasets import GPDataset
from utils.helpers import get_only_first_item


### TUTORIAL 1 ###
# DATA
def get_gp_datasets(n_samples=100000, n_points=128):
    """
    Return different 1D functions sampled from GPs with the following kernels:
    "rbf", "periodic", "non-stationary", "matern", "noisy-matern".

    Note
    ----
    - Hyper parameters chosen such that highest frequency is around 50.
    Tested empirically.
    """
    datasets = dict()
    kwargs = dict(n_samples=n_samples, n_points=n_points)
    datasets["rbf"] = GPDataset(kernel=RBF(length_scale=.1),
                                **kwargs)
    datasets["periodic"] = GPDataset(kernel=ExpSineSquared(length_scale=.3, periodicity=1.0),
                                     **kwargs)
    datasets["matern"] = GPDataset(kernel=Matern(length_scale=.1, nu=1.5),
                                   **kwargs)
    datasets["noisy-matern"] = GPDataset(kernel=(WhiteKernel(noise_level=.1) +
                                                 Matern(length_scale=.1, nu=1.5)),
                                         **kwargs)
    return datasets


def get_gp_datasets_varying(n_samples=10000, n_points=128, n_diff_kernel_hyp=100,
                            save_file=None):
    """
    Return different 1D functions sampled from GPs with the following kernels:
    "rbf", "periodic", "non-stationary", "matern", "noisy-matern" with varying
    hyperparameters.

    Note
    ----
    - Hyper parameters chosen such that highest possible frequency is below 100 =>
    can use 256 sampling rate in convolutions. Tested empirically.
    """
    datasets = dict()
    kwargs = dict(n_samples=n_samples,
                  n_points=n_points,
                  n_diff_kernel_hyp=n_diff_kernel_hyp)  # varying 1/10 per epoch

    def add_dataset_(name, kernel, save_file=save_file):
        if save_file is not None:
            save_file = (save_file, name)
        datasets[name] = GPDataset(kernel=kernel, save_file=save_file, **kwargs)

    add_dataset_("vary-rbf", RBF(length_scale_bounds=(.02, .3)))
    add_dataset_("vary-periodic",
                 ExpSineSquared(length_scale_bounds=(.2, .5), periodicity_bounds=(.5, 2.)))
    add_dataset_("vary-matern",
                 Matern(length_scale_bounds=(.03, .3), nu=1.5))
    add_dataset_("vary-noisy-matern",
                 (WhiteKernel(noise_level_bounds=(.05, .7)) +
                  Matern(length_scale_bounds=(.03, .3), nu=1.5)))

    return datasets


# MODEL
get_cntxt_trgt = CntxtTrgtGetter(contexts_getter=GetRandomIndcs(min_n_indcs=0.05, max_n_indcs=.5),
                                 targets_getter=GetRandomIndcs(min_n_indcs=0.05, max_n_indcs=.5),
                                 is_add_cntxts_to_trgts=True)  # add all context points to tagrtes

CNP_KWARGS = dict(get_cntxt_trgt=get_cntxt_trgt,
                  aggregator=torch.mean,
                  XEncoder=change_param(MLP, n_hidden_layers=1),  # share X encoding (not done in the paper)
                  XYEncoder=merge_flat_input(change_param(MLP, n_hidden_layers=2),
                                             is_sum_merge=True),  # sum the encoded X and Y
                  Decoder=merge_flat_input(change_param(MLP, n_hidden_layers=4),
                                           is_sum_merge=True),  # sum the encoded X and Y
                  r_dim=128,
                  PredictiveDistribution=Normal,  # Gaussian predictive distribution
                  encoded_path="deterministic")  # use CNP

# TRAINING


def train_all_models_(data_models, chckpnt_dirname,
                      is_retrain=False,
                      is_progress_bar=True,
                      batch_size=64,
                      max_epochs=50,
                      **kwargs):
    """Train or loads IN PLACE a dictionary containing a model and a datasets"""
    trainers = dict()
    for k, (neural_proc, dataset) in data_models.items():
        print()
        print("--- {} {} ---".format("Training" if is_retrain else "Loading", k))
        print()

        chckpt = Checkpoint(dirname=chckpnt_dirname + "_{}".format(k),
                            monitor='train_loss_best')  # train would be same as validation as always resamples

        callbacks = [chckpt]
        if is_progress_bar:
            callbacks.append(ProgressBar())
        model = NeuralNet(neural_proc, NeuralProcessLoss,
                          iterator_train__shuffle=True,  # shuffle iterator
                          train_split=None,  # don't use cross validation dev set
                          warm_start=True,  # continue training if stop and restart
                          device="cuda" if torch.cuda.is_available() else "cpu",
                          optimizer=Adam,
                          max_epochs=max_epochs,
                          batch_size=batch_size,
                          lr=1e-3,  # they use 5e-5 because 16 batch size but that would be slow
                          callbacks=callbacks,
                          **kwargs)

        if is_retrain:
            # give both X and y to `forward`
            model.fit({'X': get_only_first_item(dataset), "y": dataset.targets})

        # load in all case => even when training loads the best checkpoint
        model.initialize()
        model.load_params(checkpoint=chckpt)

        trainers[k] = model

    return trainers


def get_percentile_converge_epoch(history, percentile=0.01):
    best_loss = history[-1]['train_loss']
    init_loss = history[0]['train_loss']
    threshold = init_loss + (best_loss - init_loss) * (1 - percentile)
    for h in history:
        if h['train_loss'] < threshold:
            return h["epoch"]
