from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


import os

DIR = os.path.abspath(os.path.dirname(__file__))


class CIFAR10(datasets.CIFAR10):
    """CIFAR10 wrapper. Docs: `datasets.CIFAR10.`

    Notes
    -----
    - Transformations (and their order) follow [1] besides the fact that we scale
    the images to be in [0,1] to make it easier to use probabilistic generative models.

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test'}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """
    shape = (3, 32, 32)
    n_classes = 10

    def __init__(self,
                 root=os.path.join(DIR, '../../data/CIFAR10'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [transforms.RandomHorizontalFlip(),
                               transforms.Lambda(lambda x: random_translation(x, 2)),
                               transforms.ToTensor(),
                               # adding random noise of std 0.15 but clip to 0,1
                               transforms.Lambda(lambda x: gaussian_noise(x, std=0.15 * self.noise_factor, min=0, max=1))]
        elif split == "test":
            transforms_list = [transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         train=split == "train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        basename = os.path.join(root, "clean_{}".format(split))
        transforms_X = [global_contrast_normalization,
                        lambda x: zca_whitening(x, root, is_load=split == "test"),
                        lambda x: robust_minmax_scale(x, root, is_load=split == "test"),
                        lambda x: (x * 255).astype(np.uint8)]  # back to 255 for PIL
        self.data, self.targets = precompute_batch_tranforms(self.data, self.targets, basename,
                                                             transforms_X=transforms_X,
                                                             logger=logger)
        self.targets = to_numpy(self.targets)

        # DIRTY make sure that the noise added is also scaled
        robust_scaler = joblib.load(os.path.join(root, "robust_scaler.npy"))
        self.noise_factor = 1 / robust_scaler.data_range_[0]


class SVHN(datasets.SVHN):
    """SVHN wrapper. Docs: `datasets.SVHN.`

    Notes
    -----
    - Transformations (and their order) follow [1] besides the fact that we scale
    the images to be in [0,1] isntead of [-1,1] to make it easier to use
    probabilistic generative models.

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """
    shape = (3, 32, 32)
    n_classes = 10

    def __init__(self,
                 root=os.path.join(DIR, '../../data/SVHN'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [transforms.Lambda(lambda x: random_translation(x, 2)),
                               transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         split=split,
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        self.labels = to_numpy(self.labels)

    @property
    def targets(self):
        # make compatible with CIFAR10 dataset
        return self.labels

    @targets.setter
    def targets(self, value):
        self.labels = value


class MNIST(datasets.MNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.MNIST`.
    """
    shape = (1, 32, 32)
    n_classes = 10

    def __init__(self,
                 root=os.path.join(DIR, '../../data/MNIST'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [transforms.Resize(32),
                               transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.Resize(32),
                               transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(root,
                         train=split == "train",
                         download=True,
                         transform=transforms.Compose(transforms_list),
                         **kwargs)

        self.targets = to_numpy(self.targets)
