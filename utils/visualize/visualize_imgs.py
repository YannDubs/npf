import random

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

__all__ = ["plot_dataset_samples_imgs"]

DFLT_FIGSIZE = (17, 9)


def plot_dataset_samples_imgs(dataset, n_plots=4, figsize=DFLT_FIGSIZE, ax=None,
                              pad_value=1):
    """Plot `n_samples` samples of the a datset."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    img_tensor = torch.stack([dataset[random.randint(0, len(dataset) - 1)][0]
                              for i in range(n_plots)], dim=0)
    grid = make_grid(img_tensor,
                     nrow=2,
                     pad_value=pad_value)

    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')
