import os

import h5py
import torch


def _parse_save_file_chunk(save_file, idx_chunk):
    if save_file is None:
        save_file, save_group = None, None
    elif isinstance(save_file, tuple):
        save_file, save_group = save_file[0], save_file[1] + "/"
    elif isinstance(save_file, str):
        save_file, save_group = save_file, ""
    else:
        raise ValueError("Unsupported type of save_file={}.".format(save_file))

    if idx_chunk is not None:
        chunk_suffix = "_chunk_{}".format(idx_chunk)
    else:
        chunk_suffix = ""

    return save_file, save_group, chunk_suffix


class NotLoadedError(Exception):
    pass


def load_chunk(keys, save_file, idx_chunk):
    items = dict()
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None or not os.path.exists(save_file):
        raise NotLoadedError()

    try:
        with h5py.File(save_file, 'r') as hf:
            for k in keys:
                items[k] = torch.from_numpy(hf["{}{}{}".format(save_group, k, chunk_suffix)][:])
    except KeyError:
        raise NotLoadedError()

    return items


def save_chunk(to_save, save_file, idx_chunk, logger=None):
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None:
        return  # don't save

    if logger is not None:
        logger.info("Saving group {} chunk {} for future use ...".format(save_group, idx_chunk))

    with h5py.File(save_file, 'a') as hf:
        for k, v in to_save.items():
            hf.create_dataset("{}{}{}".format(save_group, k, chunk_suffix), data=v.numpy())
