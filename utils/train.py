from copy import deepcopy

import torch
from torch.optim import Adam

import skorch
from skorch import NeuralNet
from skorch.callbacks import ProgressBar, Checkpoint, EarlyStopping
from skorch.helper import predefined_split

from .helpers import FixRandomSeed

__all__ = ["train_models"]


def train_models(datasets, models, criterion,
                 chckpnt_dirname=None,
                 is_retrain=False,
                 train_split=skorch.dataset.CVSplit(0.1),
                 device=None,
                 max_epochs=50,
                 batch_size=64,
                 lr=1e-3,
                 optimizer=Adam,
                 callbacks=[ProgressBar()],
                 patience=None,
                 seed=None,
                 **kwargs
                 ):
    """
    Train or loads the models.

    Parameters
    ----------
    datasets : dict
        The datasets on which to train the models. If you have a train and
        validation set, the values should be a tuple (data_train, data_valid).

    models : dict
        The models to train (initialized or not). Each model will be trained on
        all datasets. If the initialzed models are passed, it will continue
        training from there.  Can also give a dictionary of dictionaries, if the
        models to train depend on the dataset.

    criterion : nn.Module
        The uninitialized criterion (loss).

    chckpnt_dirname : str, optional
        Directory where checkpoints will be saved.

    is_retrain : bool, optional
        Whether to retrain the model. If not, `chckpnt_dirname` should be given
        to load the pretrained model.

    train_split : callable, optional
        If None, there is no train/validation split. Else, train_split
        should be a function or callable that is called with X and y
        data and should return the tuple ``dataset_train, dataset_valid``.
        The validation data may be None. Use `skorch.dataset.CVSplit` to randomly
        split the data into train and validation. Only used if validation set
        not given in `datasets`.

    device : str, optional
        The compute device to be used (input to torch.device). If `None` uses
        "cuda" if available else "cpu".

    max_epochs : int, optional
        Maximum number of epochs.

    batch_size : int, optional
        Training batch size.

    lr : float, optional
        Learning rate.

    optimizer : torch.optim.Optimizer, optional
        Optimizer.

    callbacks : list, optional
        Callbacks to use.

    patience : int, optional
        Patience for early stopping. If not `None` has to be given a validation
        set.

    seed : int, optional
        Pseudo random seed to force deterministic results (on CUDA might still
        differ a little).

    kwargs :
        Additional arguments to `NeuralNet`.

    """
    trainers = dict()
    callbacks_dflt = callbacks

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if "iterator_train" not in kwargs and "iterator_train__shuffle" not in kwargs:
        # default to shuffle when using default iterator
        kwargs["iterator_train__shuffle"] = True

    if "iterator_valid" not in kwargs and "iterator_valid__batch_size" not in kwargs:
        # default to validation batch size when using default iterator
        kwargs["iterator_valid__batch_size"] = 128

    for data_name, data_train in datasets.items():

        if isinstance(list(models.values())[0], dict):
            # if dict of dict then depends on data
            current_models = models[data_name]
        else:
            current_models = models

        for model_name, model in current_models.items():

            callbacks = deepcopy(callbacks_dflt)

            if isinstance(data_train, tuple):
                data_train, data_valid = data_train
                train_split = predefined_split(data_valid)

            suffix = data_name + "/" + model_name

            print("\n--- {} {} ---\n".format("Training" if is_retrain else "Loading", suffix))

            if chckpnt_dirname is not None:
                chckpt = Checkpoint(dirname=chckpnt_dirname + suffix,
                                    monitor='valid_loss_best')
                callbacks.append(chckpt)

            if patience is not None:
                callbacks.append(EarlyStopping(patience=patience))

            if seed is not None:
                callbacks.append(FixRandomSeed(seed))

            model = NeuralNet(model, criterion,
                              train_split=train_split,
                              warm_start=True,  # continue training
                              callbacks=callbacks,
                              device=device,
                              optimizer=optimizer,
                              lr=lr,
                              max_epochs=max_epochs,
                              batch_size=batch_size,
                              **kwargs)

            if is_retrain:
                _ = model.fit(data_train)

            # load in all case => even when training loads the best checkpoint
            model.initialize()
            model.load_params(checkpoint=chckpt)

            model.module_.cpu()  # make sure on cpu
            torch.cuda.empty_cache()  # empty cache for next run

            trainers[suffix] = model

            # print best loss
            try:
                for epoch, history in enumerate(model.history[::-1]):
                    if history["valid_loss_best"]:
                        print(suffix, "best epoch:", len(model.history) - epoch,
                              "val_loss:", history["valid_loss"])
                        break
            except:
                pass

    return trainers
