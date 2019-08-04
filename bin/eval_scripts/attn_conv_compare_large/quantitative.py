import os
from os.path import dirname, abspath
base_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
os.chdir(base_dir)

import sys
sys.path.append("notebooks")
sys.path.append(".")

import json
import numpy as np
import torch
import sklearn


from neuralproc import GlobalNeuralProcess, discard_ith_arg, AttentiveNeuralProcess
from neuralproc.utils.helpers import change_param, rescale_range, MultivariateNormalDiag
from neuralproc.utils.datasplit import (get_all_indcs, CntxtTrgtGetter, GetRandomIndcs, GetRangeIndcs,
                                        GetIndcsMerger, precomputed_cntxt_trgt_split)
from neuralproc.predefined import UnetCNN, CNN, SelfAttention, MLP
from neuralproc.utils.setcnn import SetConv, MlpRBF, GaussianRBF
from neuralproc.encoders import SinusoidalEncodings
from utils.helpers import count_parameters
from utils.datasets import cntxt_trgt_precompute
from ntbks_helpers import train_all_models_, get_percentile_converge_epoch, CNP_KWARGS

from utils.visualize import plot_posterior_samples, plot_prior_samples, plot_dataset_samples
from ntbks_helpers import get_gp_datasets, get_gp_datasets_varying  # defined in first tutorial (CNP)

get_cntxt_trgt = CntxtTrgtGetter(contexts_getter=GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=.5),
                                 targets_getter=GetRandomIndcs(min_n_indcs=0.5, max_n_indcs=0.99),
                                 is_add_cntxts_to_trgts=False)  # don't context points to tagrtes

X_DIM = 1  # 1D spatial input
Y_DIM = 1  # 1D regression
N_POINTS = 128
N_SAMPLES = 100000  # this is a lot and can work with less
N_DIFF_HYP = 1000
MAX_EPOCHS = 50
DIR = "results/attn_conv_compare_large/data_1D"
datasets = get_gp_datasets_varying(n_samples=N_SAMPLES, n_points=N_POINTS,
                                   n_diff_kernel_hyp=N_DIFF_HYP, save_file='data/gp_dataset_test.hdf5')


def load_run_k(run):
    loaded = dict()

    model_name = "extended_gnp_simple"

    gnp_kwargs = dict(r_dim=64,
                      keys_to_tmp_attn=change_param(SetConv, RadialBasisFunc=MlpRBF),
                      TmpSelfAttn=change_param(CNN,
                                               Conv=torch.nn.Conv1d,
                                               n_layers=4,
                                               is_depth_separable=True,
                                               Normalization=torch.nn.Identity,
                                               is_chan_last=True,
                                               kernel_size=11),
                      tmp_to_queries_attn=change_param(SetConv, RadialBasisFunc=GaussianRBF),
                      is_skip_tmp=False,
                      is_use_x=False,
                      get_cntxt_trgt=get_cntxt_trgt,
                      is_encode_xy=False)

    # initialize one model for each dataset
    data_models = {name: (GlobalNeuralProcess(X_DIM, Y_DIM, **gnp_kwargs), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "extended_gnp_unet"

    gnp_kwargs = dict(r_dim=32,
                      keys_to_tmp_attn=change_param(SetConv, RadialBasisFunc=MlpRBF),
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
                                               kernel_size=9,
                                               max_nchannels=128,
                                               is_force_same_bottleneck=True),
                      tmp_to_queries_attn=change_param(SetConv, RadialBasisFunc=GaussianRBF),
                      is_skip_tmp=False,
                      is_use_x=False,
                      get_cntxt_trgt=get_cntxt_trgt,
                      is_encode_xy=False)

    # initialize one model for each dataset
    data_models = {name: (GlobalNeuralProcess(X_DIM, Y_DIM, **gnp_kwargs), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "extended_gnp_unet_no_share"

    gnp_kwargs = dict(r_dim=32,
                      keys_to_tmp_attn=change_param(SetConv, RadialBasisFunc=MlpRBF),
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
                                               kernel_size=9,
                                               max_nchannels=128,
                                               is_force_same_bottleneck=False),
                      tmp_to_queries_attn=change_param(SetConv, RadialBasisFunc=GaussianRBF),
                      is_skip_tmp=False,
                      is_use_x=False,
                      get_cntxt_trgt=get_cntxt_trgt,
                      is_encode_xy=False)

    # initialize one model for each dataset
    data_models = {name: (GlobalNeuralProcess(X_DIM, Y_DIM, **gnp_kwargs), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "extended_gnp_simple_only_mlp"

    gnp_kwargs = dict(r_dim=64,
                      keys_to_tmp_attn=change_param(SetConv, is_vanilla=True,
                                                    RadialBasisFunc=MlpRBF),  # onyl diff with vanilla
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

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    #loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "extended_gnp_simple_only_normalize"

    gnp_kwargs = dict(r_dim=64,
                      keys_to_tmp_attn=change_param(SetConv,
                                                    is_vanilla=False,  # onyl diff with vanilla
                                                    RadialBasisFunc=GaussianRBF),
                      TmpSelfAttn=change_param(CNN,
                                               Conv=torch.nn.Conv1d,
                                               n_layers=5,
                                               is_depth_separable=False,
                                               Normalization=torch.nn.Identity,
                                               is_chan_last=True,
                                               kernel_size=11),
                      tmp_to_queries_attn=change_param(SetConv,
                                                       is_vanilla=False,  # onyl diff with vanilla
                                                       RadialBasisFunc=GaussianRBF),
                      is_skip_tmp=False,
                      is_use_x=False,
                      get_cntxt_trgt=get_cntxt_trgt,
                      is_encode_xy=False)

    # initialize one model for each dataset
    data_models = {name: (GlobalNeuralProcess(X_DIM, Y_DIM, **gnp_kwargs), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    #loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "extended_gnp_simple_only_depthsep"

    gnp_kwargs = dict(r_dim=64,
                      keys_to_tmp_attn=change_param(SetConv, is_vanilla=True,
                                                    RadialBasisFunc=GaussianRBF),
                      TmpSelfAttn=change_param(CNN,
                                               Conv=torch.nn.Conv1d,
                                               n_layers=5,
                                               is_depth_separable=True,  # onyl diff with vanilla
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

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    #model_name = "extended_anp_sin"

    ANP_KWARGS = dict(get_cntxt_trgt=get_cntxt_trgt,
                      r_dim=128,
                      encoded_path="deterministic",  # use CNP
                      attention="weighted_dist",
                      XEncoder=SinusoidalEncodings,
                      is_relative_pos=False)

    # initialize one model for each dataset
    data_models = {name: (AttentiveNeuralProcess(X_DIM, Y_DIM, **ANP_KWARGS), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "extended_anp_rel"

    ANP_KWARGS = dict(get_cntxt_trgt=get_cntxt_trgt,
                      r_dim=128,
                      encoded_path="deterministic",  # use CNP
                      attention="transformer",
                      is_relative_pos=True)

    # initialize one model for each dataset
    data_models = {name: (AttentiveNeuralProcess(X_DIM, Y_DIM, **ANP_KWARGS), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "baseline_rbf"

    gnp_kwargs = dict(r_dim=128,
                      keys_to_tmp_attn=change_param(SetConv,
                                                    RadialBasisFunc=change_param(GaussianRBF,
                                                                                 max_dist_weight=0.7)),
                      TmpSelfAttn=None,
                      tmp_to_queries_attn=torch.nn.Identity,
                      is_skip_tmp=False,
                      is_use_x=False,
                      get_cntxt_trgt=get_cntxt_trgt,
                      is_encode_xy=False)

    # initialize one model for each dataset
    data_models = {name: (GlobalNeuralProcess(X_DIM, Y_DIM, **gnp_kwargs), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "vanilla_anp"

    anp_kwargs = dict(r_dim=128,
                      get_cntxt_trgt=get_cntxt_trgt,
                      encoded_path="deterministic",
                      attention="multihead",
                      is_relative_pos=False)

    # initialize one model for each dataset
    data_models = {name: (AttentiveNeuralProcess(X_DIM, Y_DIM, **anp_kwargs), data)
                   for name, data in datasets.items()}

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    model_name = "vanilla_gnp"

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

    trainers = train_all_models_(data_models,
                                 "{}/run_k{}/{}".format(DIR, run, model_name),
                                 is_retrain=False)  # if false load precomputed

    loaded[model_name] = dict(data_models=data_models, trainers=trainers)

    return loaded


def score_cnp(trainer, dataset, get_cntxt_trgt=None, save_file=None,
              batch_size=64, n_test=100000, **kwargs):
    """Return the log likelihood"""
    score = 0
    n_steps = 0
    get_cntxt_trgt_old = trainer.module_.get_cntxt_trgt

    if get_cntxt_trgt is None:
        get_cntxt_trgt = trainer.module_.get_cntxt_trgt

    # when evealuating loss, use all targtes in that are not in context (but still sampel context)
    get_cntxt_trgt.set_eval()

    # use th eloss as metric (i.e. return log likelihood)
    trainer.criterion_.is_use_as_metric = True

    trainer.module_.get_cntxt_trgt = precomputed_cntxt_trgt_split

    for i in range(0, n_test // batch_size):
        n_steps += 1

        # only get sample from data if not already precomputed
        (X_cntxt, Y_cntxt, X_trgt, Y_trgt
         ) = cntxt_trgt_precompute(lambda: dataset.get_samples(n_samples=batch_size, **kwargs),
                                   get_cntxt_trgt, save_file,
                                   idx_chunk=i)

        # puts in a skorch format (because y is splitted in the module itself)
        step = trainer.validation_step({"X": X_cntxt, "y": Y_cntxt, "X_trgt": X_trgt, "y_trgt": Y_trgt}, Y_trgt)
        score += step["loss"].item()

    score /= n_steps

    get_cntxt_trgt.reset()  # reset in case used in future

    trainer.module_.get_cntxt_trgt = get_cntxt_trgt_old

    return score


def score_cnp_extrap(trainer, dataset, extrap_mode, extrap_distance=4, **kwargs):
    """Return the log likelihood of extrapolation."""
    interpolation_range = dataset.min_max
    extrapolation_range = (dataset.min_max[0], dataset.min_max[1] + extrap_distance)

    extrap_rescaled_range = tuple(rescale_range(np.array(extrapolation_range), interpolation_range, (-1, 1)))
    trainer.module_.set_extrapolation(extrap_rescaled_range)  # set the model in extrapolation mode

    train_n_points = dataset.n_points
    total_n_points = int(rescale_range(dataset.n_points, interpolation_range, extrapolation_range))

    if extrap_mode == "shift":
        # shifting both the targets and context equally
        targets_getter = GetRangeIndcs((train_n_points, total_n_points))
        contexts_getter = GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=0.5,
                                         range_indcs=(train_n_points, total_n_points))

    elif extrap_mode == "scale":
        # adding more context (making sure that keeping correct density for comparaison)
        contexts_getter = GetIndcsMerger([GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=0.5,
                                                         range_indcs=(0, train_n_points)),
                                          GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=0.5,
                                                         range_indcs=(train_n_points, total_n_points))])
        targets_getter = GetRangeIndcs((0, train_n_points))

    elif extrap_mode == "future":
        # predicting in the future => keeping same context but shifting targt
        contexts_getter = GetRandomIndcs(min_n_indcs=0.01, max_n_indcs=.5, range_indcs=(0, train_n_points))
        targets_getter = GetRangeIndcs((train_n_points, total_n_points))

    else:
        raise ValueError("Unkown extrap_mode = {}.".format(extrap_mode))

    get_cntxt_trgt = CntxtTrgtGetter(contexts_getter=contexts_getter,
                                     targets_getter=targets_getter,
                                     is_add_cntxts_to_trgts=False)

    score = score_cnp(trainer, dataset, n_points=total_n_points, test_min_max=extrapolation_range,
                      get_cntxt_trgt=get_cntxt_trgt, **kwargs)

    trainer.module_.set_extrapolation(interpolation_range)  # put back to normal

    return score


def score_cnp_dense(trainer, dataset, **kwargs):
    """Return the log likelihood when increasing density."""

    # making sure that have high density by taking large context
    trainer.module_.get_cntxt_trgt.tmp_args["contexts_getter"] = GetRandomIndcs(min_n_indcs=.3, max_n_indcs=.7)

    # increase density by 100 and decrease test number by 10 (if not time ++), also decrease batchsize
    # for memory reasons
    score = score_cnp(trainer, dataset, n_points=dataset.n_points * 10, n_test=50000, batch_size=32, **kwargs)

    return score


def make_all_summaries(loaded, save_file=None):
    """Return a dictionary of summaries: group -> data -> summary."""
    all_summaries = dict()

    for i, (model_name, loaded_group) in enumerate(loaded.items()):
        print()
        print("--- {} ---".format(model_name))
        all_summaries[model_name] = dict()
        for data_name, trainer in loaded_group["trainers"].items():
            print(data_name)
            history = trainer.history
            dataset = loaded_group["data_models"][data_name][1]

            converged_epoch = len(history)
            train_log_likelihood = - history[-1]["train_loss"]
            train_losses = [h['train_loss'] for h in history]
            percentile_converged_epoch = get_percentile_converge_epoch(history, percentile=0.1)
            time_per_epochs = sum(h['dur'] for h in history) / len(history)

            test_log_likelihood_interp = score_cnp(trainer, dataset,
                                                   save_file=("data/gp_dataset_test.hdf5", "{}/test_interp".format(data_name)))
            test_log_likelihood_future = score_cnp_extrap(trainer, dataset, "future",
                                                          save_file=("data/gp_dataset_test.hdf5", "{}/test_future".format(data_name)))
            test_log_likelihood_scale = score_cnp_extrap(trainer, dataset, "scale",
                                                         save_file=("data/gp_dataset_test.hdf5", "{}/test_scale".format(data_name)))
            test_log_likelihood_shift = score_cnp_extrap(trainer, dataset, "shift",
                                                         save_file=("data/gp_dataset_test.hdf5", "{}/test_shift".format(data_name)))

            try:
                test_log_likelihood_dense = score_cnp_dense(trainer, dataset,
                                                            save_file=("data/gp_dataset_test.hdf5", "{}/test_dense".format(data_name)))
            except:
                print("issue with dense for", data_name)
                test_log_likelihood_dense = float("inf")

            all_summaries[model_name][data_name] = dict(converged_epoch=converged_epoch,
                                                        train_log_likelihood=train_log_likelihood,
                                                        percentile_converged_epoch=percentile_converged_epoch,
                                                        time_per_epochs=time_per_epochs,
                                                        test_log_likelihood_interp=test_log_likelihood_interp,
                                                        test_log_likelihood_future=test_log_likelihood_future,
                                                        test_log_likelihood_scale=test_log_likelihood_scale,
                                                        test_log_likelihood_shift=test_log_likelihood_shift,
                                                        test_log_likelihood_dense=test_log_likelihood_dense,
                                                        train_losses=train_losses)

    if save_file is not None:
        if os.path.exists(save_file):
            os.rename(save_file, save_file + ".bak")

        with open(save_file, "w") as f:
            json.dump(all_summaries, f)

    return all_summaries


def score_generator(dataset, get_cntxt_trgt=None, save_file=None, n_test=100000,
                    batch_size=64, **kwargs):
    """Return the log likelihood"""
    score = 0
    score_clamped = 0
    generator = sklearn.base.clone(dataset.generator)

    # the intial param will be the correct value, which is cheating => randomly chose a new plausible intiial value
    for hyperparam in generator.kernel.hyperparameters:
        generator.kernel.set_params(**{hyperparam.name: np.random.uniform(*hyperparam.bounds.squeeze())})

    if get_cntxt_trgt is not None:
        # when evealuating loss, use all targtes in that are not in context (but still sampel context)
        get_cntxt_trgt.set_eval()

    for i in range(n_test // batch_size):
        # only get sample from data if not already precomputed
        (X_cntxt, Y_cntxt, X_trgt, Y_trgt
         ) = cntxt_trgt_precompute(lambda: dataset.get_samples(n_samples=batch_size, **kwargs),
                                   get_cntxt_trgt, save_file,
                                   idx_chunk=i)

        # for generator should not be in -1 1
        X_cntxt = rescale_range(X_cntxt, (-1, 1), dataset.min_max).numpy()
        X_trgt = rescale_range(X_trgt, (-1, 1), dataset.min_max).numpy()
        Y_cntxt = Y_cntxt.numpy()
        Y_trgt = Y_trgt.double()  # double precison for loss and leave in pytorch

        for j in range(1):  # just do 1 per batch if not would be way too long

            Xi_cntxt, Yi_cntxt, Xi_trgt, Yi_trgt = X_cntxt[j], Y_cntxt[j], X_trgt[j], Y_trgt[j]
            generator.fit(Xi_cntxt, Yi_cntxt)
            mean_y, std_y = generator.predict(Xi_trgt, return_std=True)
            # use exact same loss as for CNP (could use scipy.stats.multivariate_normal.logpdf) but results differef
            m = MultivariateNormalDiag(torch.from_numpy(mean_y), torch.from_numpy(std_y))
            m_clamped = MultivariateNormalDiag(torch.from_numpy(mean_y), torch.from_numpy(std_y).clamp(min=0.1))
            score += m.log_prob(Yi_trgt).mean().item()
            score_clamped += m_clamped.log_prob(Yi_trgt).mean().item()

    score /= n_test
    score_clamped /= n_test

    if get_cntxt_trgt is not None:
        get_cntxt_trgt.reset()  # reset in case used in future

    return score, score_clamped


def add_generator_results(all_summaries, datasets, save_file=None):
    print("Computing Genrator results ...")
    all_summaries["Generator"] = dict()

    for k, dataset in datasets.items():
        test_log_likelihood_interp = score_generator(dataset, save_file=("data/gp_dataset_test.hdf5", "{}/test_interp".format(k)))
        test_log_likelihood_future = score_generator(dataset, save_file=("data/gp_dataset_test.hdf5", "{}/test_future".format(k)))
        test_log_likelihood_scale = score_generator(dataset, save_file=("data/gp_dataset_test.hdf5", "{}/test_scale".format(k)))
        test_log_likelihood_shift = score_generator(dataset, save_file=("data/gp_dataset_test.hdf5", "{}/test_shift".format(k)))
        test_log_likelihood_dense = score_generator(dataset, save_file=("data/gp_dataset_test.hdf5", "{}/test_dense".format(k)))

        all_summaries["Generator"][k] = dict(
            test_log_likelihood_interp=test_log_likelihood_interp[0],
            test_log_likelihood_interp_clamped=test_log_likelihood_interp[1],
            test_log_likelihood_future=test_log_likelihood_future[0],
            test_log_likelihood_scale=test_log_likelihood_scale[0],
            test_log_likelihood_shift=test_log_likelihood_shift[0],
            test_log_likelihood_dense=test_log_likelihood_dense[0],
            test_log_likelihood_future_clamped=test_log_likelihood_future[1],
            test_log_likelihood_scale_clamped=test_log_likelihood_scale[1],
            test_log_likelihood_shift_clamped=test_log_likelihood_shift[1],
            test_log_likelihood_dense_clamped=test_log_likelihood_dense[1])

    if save_file is not None:
        if os.path.exists(save_file):
            os.rename(save_file, save_file + ".bak")

    with open(save_file, "w") as f:
        json.dump(all_summaries, f)

    return all_summaries


def summarise_run_k(run, basename="results/attn_conv_compare_large/summaries"):

    loaded = load_run_k(run)
    print("MAKING SUMMARIES")
    all_summaries = make_all_summaries(loaded,
                                       save_file="{}_run_k{}.json".format(basename, k))
    """
    with open("{}_run_k{}.json".format(basename, k), "r") as f:
        all_summaries = json.load(f)


    all_summaries = add_generator_results(all_summaries, datasets,
                                          save_file="{}_run_k{}_with_gen.json".format(basename, k))
    """


if __name__ == "__main__":
    for k in range(1, 3):  # onyl one iter for now
        summarise_run_k(k)
