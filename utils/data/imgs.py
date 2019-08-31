import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from skorch.utils import to_numpy

from .helpers import random_translation, preprocess

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = torch.tensor([0., 0., 0.])
COLOUR_WHITE = torch.tensor([1., 1., 1.])
COLOUR_BLUE = torch.tensor([0., 0., 1.])
DATASETS_DICT = {"mnist": "MNIST",
                 "svhn": "SVHN",
                 "celeba32": "CelebA32",
                 "celeba64": "CelebA64"}
DATASETS = list(DATASETS_DICT.keys())


# HELPERS

def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).shape


# TORCHVISION DATASETS
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
    missing_px_color = COLOUR_BLACK
    n_classes = 10

    def __init__(self,
                 root=os.path.join(DIR, '../../data/SVHN'),
                 split="train",
                 logger=logging.getLogger(__name__),
                 **kwargs):

        if split == "train":
            transforms_list = [  # transforms.Lambda(lambda x: random_translation(x, 2)),
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
    missing_px_color = COLOUR_BLUE

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


# EXTERNAL DATASETS
class ExternalDataset(Dataset, abc.ABC):
    """Base Class for external datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class CelebA64(ExternalDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    shape = (3, 64, 64)
    missing_px_color = COLOUR_BLACK
    n_classes = 0  # not classification

    def __init__(self, root=os.path.join(DIR, '../../data/celeba64'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root)

        try:
            subprocess.check_call(["curl", "-L", type(self).urls["train"],
                                   "--output", save_path])
        except FileNotFoundError as e:
            raise Exception(e + " Please instal curl with `apt-get install curl`...")

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).shape[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = plt.imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class CelebA32(CelebA64):
    shape = (3, 32, 32)

    def __init__(self, root=os.path.join(DIR, '../../data/celeba32'), **kwargs):
        super().__init__(root, *kwargs)

        self.imgs = glob.glob(self.train_data + '/*')
