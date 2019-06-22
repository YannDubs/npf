import random
import functools

import torch
import numpy as np

from .helpers import ratio_to_int, prod, indep_shuffle_


### INDICES SELECTORS ###
def get_all_indcs(batch_size, n_possible_points):
    """
    Return all possible indices.
    """
    return torch.arange(n_possible_points).expand(batch_size, n_possible_points)


class GetRandomIndcs:
    """
    Return random subset of indices.

    Parameters
    ----------
    min_n_indcs : float or int, optional
        Minimum number of indices. If smaller than 1, represents a percentage of
        points.

    max_n_indcs : float or int, optional
        Maximum number of indices. If smaller than 1, represents a percentage of
        points.

    is_batch_repeat : bool, optional
        Whether to use use the same indices for all elements in the batch.
    """

    def __init__(self,
                 min_n_indcs=.1,
                 max_n_indcs=.5,
                 is_batch_repeat=False):
        self.min_n_indcs = min_n_indcs
        self.max_n_indcs = max_n_indcs
        self.is_batch_repeat = is_batch_repeat

    def __call__(self, batch_size, n_possible_points):
        min_n_indcs = ratio_to_int(self.min_n_indcs, n_possible_points)
        max_n_indcs = ratio_to_int(self.max_n_indcs, n_possible_points)
        # make sure select at least 1
        n_indcs = random.randint(max(1, min_n_indcs), max(1, max_n_indcs))

        if self.is_batch_repeat:
            indcs = torch.randperm(n_possible_points)[:n_indcs]
            indcs = indcs.unsqueeze(0).expand(batch_size, n_indcs)
        else:
            indcs = np.arange(n_possible_points
                              ).reshape(1, n_possible_points
                                        ).repeat(batch_size, axis=0)
            indep_shuffle_(indcs, -1)
            indcs = torch.from_numpy(indcs[:, :n_indcs])

        return indcs


class CntxtTrgtGetter:
    """
    Split a dataset into context and target points based on indices.

    Parameters
    ----------
    contexts_getter : callable, optional
        Get the context indices if not given directly (useful for training).

    targets_masker : callable, optional
        Get the context indices if not given directly (useful for training).

    is_add_cntxts_to_trgts : bool, optional
        Whether to add the context points to the targets.
    """

    def __init__(self,
                 contexts_getter=GetRandomIndcs(),
                 targets_getter=get_all_indcs,
                 is_add_cntxts_to_trgts=True):
        self.contexts_getter = contexts_getter
        self.targets_getter = targets_getter
        self.is_add_cntxts_to_trgts = is_add_cntxts_to_trgts

    def __call__(self, X, y=None, context_indcs=None, target_indcs=None):
        """
        Parameters
        ----------
        X: torch.Tensor, size = [batch_size, num_points, x_dim]
            Position features. Values should always be in [-1, 1].

        Y: torch.Tensor, size = [batch_size, num_points, y_dim]
            Targets.

        context_indcs : np.array, size=[batch_size, n_indcs]
            Indices of the context points. If `None` generates it using
            `contexts_getter(batch_size, num_points)`.

        target_indcs : np.array, size=[batch_size, n_indcs]
            Indices of the target points. If `None` generates it using
            `contexts_getter(batch_size, num_points)`.
        """
        if context_indcs is None:
            context_indcs = self.contexts_getter(*self.getter_inputs(X))
        if target_indcs is None:
            target_indcs = self.targets_getter(*self.getter_inputs(X))

        if self.is_add_cntxts_to_trgts:
            target_indcs = self.add_cntxts_to_trgts(X, target_indcs, context_indcs)

        X_cntxt, Y_cntxt = self.select(X, y, context_indcs)
        X_trgt, Y_trgt = self.select(X, y, target_indcs)
        return X_cntxt, Y_cntxt, X_trgt, Y_trgt

    def add_cntxts_to_trgts(self, X, target_indcs, context_indcs):
        """
        Add context points to targets: has been shown emperically better.
        Note that this might results in duplicate indices in the targets.
        """
        num_points = X.size(1)
        target_indcs = torch.cat([target_indcs, context_indcs], dim=-1)
        # to reduce the probability of duplicating indices remove context indices
        # that made target indices larger than n_possible_points
        return target_indcs[:, :num_points]

    def getter_inputs(self, X):
        """Make the input for the getters."""
        batch_size, num_points, x_dim = X.shape
        return batch_size, num_points

    def select(self, X, y, indcs):
        """Select the correct values from X."""
        batch_size, num_points, x_dim = X.shape
        indcs = indcs.to(X.device).unsqueeze(-1).expand(batch_size, -1, x_dim)
        return torch.gather(X, 1, indcs).contiguous(), torch.gather(y, 1, indcs).contiguous()


### GRID AND MASKING ###
class RandomMasker(GetRandomIndcs):
    """
    Return random subset mask.

    Parameters
    ----------
    min_nnz : float or int, optional
        Minimum number of non zero values. If smaller than 1, represents a
        percentage of points.

    max_nnz : float or int, optional
        Maximum number of non zero values. If smaller than 1, represents a
        percentage of points.

    is_batch_repeat : bool, optional
        Whether to use use the same indices for all elements in the batch.
    """

    def __init__(self,
                 min_nnz=.1,
                 max_nnz=.5,
                 is_batch_repeat=False):
        super().__init__(min_n_indcs=min_nnz,
                         max_n_indcs=max_nnz,
                         is_batch_repeat=is_batch_repeat)

    def __call__(self, batch_size, n_possible_points):
        n_possible_points = prod(mask_shape)
        nnz_indcs = super().__call__(batch_size, n_possible_points)

        if is_batch_repeat:
            # share memory
            mask = torch.zeros(n_possible_points).byte()
            mask = mask.unsqueeze(0).expand(batch_size, n_possible_points)
        else:
            mask = torch.zeros((batch_size, n_possible_points)).byte()

        mask.scatter_(1, nnz_indcs, 1)
        mask = mask.view(batch_size, *mask_shape).contiguous()

        return mask


def and_masks(*masks):
    """Composes tuple of masks by an and operation."""
    mask = functools.reduce(lambda a, b: a & b, masks)
    return mask


def or_masks(*masks):
    """Composes tuple of masks by an or operation."""
    mask = functools.reduce(lambda a, b: a | b, masks)
    return mask


def half_masker(batch_size, mask_shape, dim=0):
    """Return a mask which masks the top half features of `dim`."""
    mask = torch.zeros(mask_shape).byte()
    slcs = [slice(None)] * (len(mask_shape))
    slcs[dim] = slice(0, mask_shape[dim] // 2)
    mask[slcs] = 1
    # share memory
    return mask.unsqueeze(0).expand(batch_size, *mask_shape)


def no_masker(batch_size, mask_shape):
    """Return a mask of all 1."""
    mask = torch.ones(1).byte()
    # share memory
    return mask.expand(batch_size, *mask_shape)


class GridCntxtTrgtGetter(CntxtTrgtGetter):
    """
    Split grids of values (e.g. images) into context and target points.

    Parameters
    ----------
    context_masker : callable, optional
        Get the context masks if not given directly (useful for training).

    target_masker : callable, optional
        Get the context masks if not given directly (useful for training).

    kwargs:
        Additional arguments to `CntxtTrgtGetter`.
    """

    def __init__(self,
                 context_masker=RandomMasker(),
                 target_masker=no_masker,
                 **kwargs):
        super().__init__(contexts_getter=context_masker,
                         targets_getter=target_masker,
                         **kwargs)

    def __call__(self, X, y=None, context_mask=None, target_mask=None):
        """
        Parameters
        ----------
        X: torch.Tensor, size=[batch_size, y_dim, *grid_shape]
            Grid input. Batch where the first dimension represents the number
            of outputs, the rest are the grid dimensions. E.g. (3, 32, 32)
            for images would mean output is 3 dimensional (channels) while
            features are on a 2d grid.

        y: None
            Placeholder

        context_mask : torch.ByteTensor, size=[batch_size, *grid_shape]
            Binary mask indicating the context. Number of non zero should
            be same for all batch. If `None` generates it using
            `context_masker(batch_size, mask_shape)`.

        target_mask : torch.ByteTensor, size=[batch_size, *grid_shape]
            Binary mask indicating the targets. Number of non zero should
            be same for all batch. If `None` generates it using
            `target_masker(batch_size, mask_shape)`.
        """
        return super().__call__(X,
                                context_indcs=context_mask,
                                target_indcs=target_mask)

    def add_cntxts_to_trgts(self, X, target_mask, context_mask):
        """Add context points to targets: has been shown emperically better."""
        return or_masks(target_mask, context_mask)

    def getter_inputs(self, X):
        """Make the input for the getters."""
        batch_size, y_dim, *grid_shape = X.shape
        return batch_size, grid_shape

    def select(self, X, y, mask):
        """
        Applies a batch of mask to a grid of size=[batch_size, y_dim, *grid_shape],
        and return a the masked X and Y values. `y` is a placeholder.
        """
        batch_size, y_dim, *grid_shape = X.shape
        n_grid_dim = len(grid_shape)

        # make sure on correct device
        device = X.device
        mask = mask.to(device)

        # batch_size, x_dim
        nonzero_idcs = mask.nonzero()
        # assume same amount of nonzero across batch
        n_cntxt = mask[0].nonzero().size(0)

        # first dim is batch idx.
        X_masked = nonzero_idcs[:, 1:].view(batch_size, n_cntxt, n_grid_dim).float()
        # normalize grid idcs to [-1,1]
        for i, size in enumerate(grid_shape):
            X_masked[:, :, i] *= 2 / (size - 1)  # in [0,2]
            X_masked[:, :, i] -= 1  # in [-1,1]

        mask = mask.unsqueeze(1).expand(batch_size, y_dim, *grid_shape)
        Y_masked = X[mask].view(batch_size, y_dim, n_cntxt)
        # batch_size, n_cntxt, self.y_dim
        Y_masked = Y_masked.permute(0, 2, 1)

        return X_masked.contiguous(), Y_masked.contiguous()
