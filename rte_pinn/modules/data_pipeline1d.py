import torch
import torch.utils.data as data
from ml_collections import ConfigDict


class Dataset(data.IterableDataset):
    def __init__(self, config: ConfigDict):
        self.domain = config.domain
        self.dataset = config.model.dataset
        self.xmin, self.xmax = self.domain.x
        self.vmin, self.vmax = self.domain.v
        self.interior_samples = self.dataset.interior_samples
        self.boundary_left_samples = self.dataset.boundary_left_samples
        self.boundary_right_samples = self.dataset.boundary_right_samples

    def __iter__(self):
        return self

    def __next__(self):
        tensor_dict = {
            "interior": {
                "x": self.xmin
                + torch.rand((self.interior_samples, 1))
                * (self.xmax - self.xmin),
                "v": self.vmin
                + torch.rand((self.interior_samples, 1))
                * (self.vmax - self.vmin),
            },
            "boundary_left": {
                "x": self.xmin * torch.ones((self.boundary_left_samples, 1)),
                "v": self.vmin
                + torch.rand((self.boundary_left_samples, 1))
                * (self.vmax - self.vmin),
            },
            "boundary_right": {
                "x": self.xmax * torch.ones((self.boundary_right_samples, 1)),
                "v": self.vmin
                + torch.rand((self.boundary_right_samples, 1))
                * (self.vmax - self.vmin),
            },
        }
        return tensor_dict


def Data_Loader(config):
    dataset = Dataset(config)
    dataloader = data.DataLoader(dataset, batch_size=None, num_workers=0)
    return iter(dataloader)
