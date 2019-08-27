from functools import partial
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn

from neuralproc.utils.helpers import rescale_range

from .helpers import NotLoadedError, load_chunk, save_chunk


__all__ = ["GPDataset"]

logging.basicConfig(level=logging.INFO)


class GPDataset(Dataset):
    """
    Dataset of functions generated by a gaussian process.

    Parameters
    ----------
    kernel : sklearn.gaussian_process.kernels or list
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default.

    min_max : tuple of floats, optional
        Min and max point at which to evaluate the function (bounds).

    n_samples : int, optional
        Number of sampled functions contained in dataset.

    n_points : int, optional
        Number of points at which to evaluate f(x) for x in min_max.

    is_vary_kernel_hyp : bool, optional
        Whether to sample each example from a kernel with random hyperparameters,
        that are sampled uniformly in the kernel hyperparameters `*_bounds`.

    save_file : string or tuple of strings, optional
        Where to save and load the dataset. If tuple `(file, group)`, save in
        the hdf5 under the given group. If `None` regenerate samples indefinitely.
        Note that if the saved dataset has been completely used,
        it will generate a new sub-dataset for every epoch and save it for future
        use.

    n_same_samples : int, optional
        Mumber of samples with same kernel hyperparameters and X. This makes the
        sampling quicker.

    kwargs:
        Additional arguments to `GaussianProcessRegressor`.
    """

    def __init__(self,
                 kernel=(WhiteKernel(noise_level=.1, noise_level_bounds=(.1, .5)) +
                         RBF(length_scale=.4, length_scale_bounds=(.1, 1.))),
                 min_max=(-2, 2),
                 n_samples=1000,
                 n_points=128,
                 is_vary_kernel_hyp=False,
                 save_file=None,
                 logging_level=logging.INFO,
                 n_same_samples=20,
                 **kwargs):

        self.n_samples = n_samples
        self.n_points = n_points
        self.min_max = min_max
        self.is_vary_kernel_hyp = is_vary_kernel_hyp
        self.logger = logging.getLogger('GPDataset')
        self.logger.setLevel(logging_level)
        self.save_file = save_file
        self.n_same_samples = n_same_samples

        self._idx_precompute = 0  # current index of precomputed data
        self._idx_chunk = 0  # current chunk (i.e. epoch)

        if not is_vary_kernel_hyp:
            # only fit hyperparam when predicting if using various hyperparam
            kwargs["optimizer"] = None
        self.generator = GaussianProcessRegressor(kernel=kernel,
                                                  alpha=0.005,  # numerical stability for preds
                                                  **kwargs)
        self.precompute_chunk_()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # doesn't use index because randomly gnerated in any case => sample
        # in order which enables to know when epoch is finished and regenerate
        # new functions
        self._idx_precompute += 1
        if self._idx_precompute == self.n_samples:
            # self._idx_precompute = 0  # DEV
            self.precompute_chunk_()
        return self.data[self._idx_precompute], self.targets[self._idx_precompute]

    def get_samples(self, n_samples=None, test_min_max=None,
                    n_points=None, save_file=None, idx_chunk=None):
        """Return a batch of samples

        Parameters
        ----------
        n_samples : int, optional
            Number of sampled function (i.e. batch size). Has to be dividable
            by n_diff_kernel_hyp or 1. If `None` uses `self.n_samples`.

        test_min_max : float, optional
            Testing range. If `None` uses training one.

        n_points : int, optional
            Number of points at which to evaluate f(x) for x in min_max. If None
            uses `self.n_points`.

        save_file : string or tuple of strings, optional
            Where to save and load the dataset. If tuple `(file, group)`, save in
            the hdf5 under the given group. If `None` uses does not save.

        idx_chunk : int, optional
            Index of the current chunk. This is used when `save_file` is not None,
            and you want to save a single dataset through multiple calls to
            `get_samples`.
        """
        test_min_max = test_min_max if test_min_max is not None else self.min_max
        n_points = n_points if n_points is not None else self.n_points
        n_samples = n_samples if n_samples is not None else self.n_samples

        try:
            loaded = load_chunk({"data", "targets"}, save_file, idx_chunk)
            data, targets = loaded["data"], loaded["targets"]
        except NotLoadedError:
            X = self._sample_features(test_min_max, n_points, n_samples)
            X, targets = self._sample_targets(X, n_samples)
            data = self._postprocessing_features(X, n_samples)
            save_chunk({"data": data, "targets": targets}, save_file, idx_chunk,
                       logger=self.logger)

        return data, targets

    def precompute_chunk_(self):
        """Load or precompute and save a chunk (data for an epoch.)"""

        self._idx_precompute = 0
        self.data, self.targets = self.get_samples(save_file=self.save_file,
                                                   idx_chunk=self._idx_chunk)
        self._idx_chunk += 1

    def _sample_features(self, min_max, n_points, n_samples):
        """Sample X with non uniform intervals. """
        X = np.random.uniform(min_max[1], min_max[0], size=(n_samples, n_points))
        # sort which is convenient for plotting
        X.sort(axis=-1)
        return X

    def _postprocessing_features(self, X, n_samples):
        """Convert the features to a tensor, rescale them to [-1,1] and expand."""
        X = torch.from_numpy(X).unsqueeze(-1).float()
        X = rescale_range(X, self.min_max, (-1, 1))
        return X

    def _sample_targets(self, X, n_samples):
        targets = X.copy()
        n_samples, n_points = X.shape
        for i in range(0, n_samples, self.n_same_samples):
            if self.is_vary_kernel_hyp:
                self.sample_kernel_()

            for attempt in range(self.n_same_samples):
                # can have numerical issues => retry using a different X
                try:
                    # takes care of boundaries
                    n_same_samples = targets[i:i + self.n_same_samples, :].shape[0]
                    targets[i:i + self.n_same_samples,
                            :] = self.generator.sample_y(X[i + attempt, :, np.newaxis],
                                                         n_samples=n_same_samples,
                                                         random_state=None
                                                         ).transpose(1, 0)
                    X[i:i + self.n_same_samples, :] = X[i + attempt, :]
                except np.linalg.LinAlgError:
                    continue  # try again
                else:
                    break  # success
            else:
                raise np.linalg.LinAlgError("SVD did not converge 10 times in a row.")

        # shuffle output to not have n_same_samples consecutive
        X, targets = sklearn.utils.shuffle(X, targets)
        targets = torch.from_numpy(targets)
        targets = targets.view(n_samples, n_points, 1).float()
        return X, targets

    def sample_kernel_(self):
        """
        Modify inplace the kernel hyperparameters through uniform sampling in their
        respective bounds.
        """
        K = self.generator.kernel
        for hyperparam in K.hyperparameters:
            K.set_params(**{hyperparam.name: np.random.uniform(*hyperparam.bounds.squeeze())})
