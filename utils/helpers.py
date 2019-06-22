import torch


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum([p.numel() for p in model.parameters()])


def get_only_first_item(to_index):
    """Helper function to make a class `to_index` return `to_index[i][0]` when indexed."""
    class FirstIndex:
        def __init__(self, to_index):
            self.to_index = to_index

        def __getitem__(self, i):
            return self.to_index[i][0]

        def __len__(self):
            return len(self.to_index)

    return FirstIndex(to_index)


def make_Xy_input(dataset, y=None):
    """
    Transform a dataset X to a variable that can be directly used like so:
    `NeuralNetEstimator.fit(*make_Xy_input(dataset))` when both `X` and `y`
    should be inputs to `forward`. Can also give a X and y.
    """
    if isinstance(dataset, dict):
        y = dataset["y"]
        X = dataset["X"]
    elif isinstance(dataset, torch.utils.data.Dataset):
        if y is None:
            try:
                y = dataset.targets
            except AttributeError:
                y = dataset.y  # skorch datasets
        X = get_only_first_item(dataset)
    else:
        # array-like or tensor
        X = dataset

    return ({'X': X, "y": y}, y)
