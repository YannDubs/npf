from functools import reduce
import operator

import numpy as np


def indep_shuffle_(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.

    Credits : https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])


def ratio_to_int(percentage, max_val):
    """Converts a ratio to an integer if it is smaller than 1."""
    if 1 <= percentage <= max_val:
        out = percentage
    elif 0 <= percentage < 1:
        out = percentage * max_val
    else:
        raise ValueError("percentage={} outside of [0,{}].".format(percentage, max_val))

    return int(out)


def prod(iterable):
    """Compute the product of all elements in an iterable."""
    return reduce(operator.mul, iterable, 1)


def rescale_range(X, old_range, new_range):
    """Rescale X linearly to be in `new_range` rather than `old_range`."""
    old_min = old_range[0]
    new_min = new_range[0]
    old_delta = old_range[1] - old_min
    new_delta = new_range[1] - new_min
    return (((X - old_min) * new_delta) / old_delta) + new_min
