import logging

import numpy as np

from .helpers import NotLoadedError, load_chunk, save_chunk


__all__ = ["cntxt_trgt_precompute"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cntxt_trgt_precompute(get_xy, get_cntxt_trgt, save_file, idx_chunk=None, **kwargs):
    try:
        loaded = load_chunk({"X_cntxt", "Y_cntxt", "X_trgt", "Y_trgt"}, save_file, idx_chunk)
        X_cntxt, Y_cntxt, X_trgt, Y_trgt = (loaded["X_cntxt"], loaded["Y_cntxt"],
                                            loaded["X_trgt"], loaded["Y_trgt"])
    except NotLoadedError:
        try_step = 0
        is_data = False

        while not is_data:
            try:
                X, y = get_xy()
                is_data = True
            except np.linalg.LinAlgError:
                try_step += 1
                if try_step > 10:
                    raise np.linalg.LinAlgError

        X_cntxt, Y_cntxt, X_trgt, Y_trgt = get_cntxt_trgt(X, y, **kwargs)

        save_chunk({"X_cntxt": X_cntxt, "Y_cntxt": Y_cntxt, "X_trgt": X_trgt, "Y_trgt": Y_trgt},
                   save_file, idx_chunk, logger=logger)

    return X_cntxt, Y_cntxt, X_trgt, Y_trgt
