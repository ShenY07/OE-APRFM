import torch
import torch.utils.data as data
from ml_collections import ConfigDict


class Dataset(data.IterableDataset):
    def __init__(self, config: ConfigDict):
        self.domain = config.domain
        self.dataset = config.model.dataset
        self.xmin, self.xmax = self.domain.x
        self.ymin, self.ymax = self.domain.y
        self.thetamin, self.thetamax = self.domain.theta
        self.interior_samples = self.dataset.interior_samples
        self.boundary_left_samples = self.dataset.boundary_left_samples
        self.boundary_right_samples = self.dataset.boundary_right_samples
        self.boundary_bottom_samples = self.dataset.boundary_bottom_samples
        self.boundary_top_samples = self.dataset.boundary_top_samples

    def __iter__(self):
        return self

    def __next__(self):
        tensor_dict = {
            "interior": {
                "x": self.xmin
                + torch.rand((self.interior_samples, 1))
                * (self.xmax - self.xmin),
                "y": self.ymin
                + torch.rand((self.interior_samples, 1))
                * (self.ymax - self.ymin),
                "theta": self.thetamin
                + torch.rand((self.interior_samples, 1))
                * (self.thetamax - self.thetamin),
            },
            "boundary_left": {
                "x": self.xmin * torch.ones((self.boundary_left_samples, 1)),
                "y": self.ymin
                + torch.rand((self.boundary_left_samples, 1))
                * (self.ymax - self.ymin),
                "theta": self.thetamin
                + torch.rand((self.boundary_left_samples, 1))
                * (self.thetamax - self.thetamin),
                # "theta": -torch.pi/2 + torch.rand((self.boundary_left_samples, 1)) * torch.pi,
            },
            "boundary_right": {
                "x": self.xmax * torch.ones((self.boundary_right_samples, 1)),
                "y": self.ymin
                + torch.rand((self.boundary_right_samples, 1))
                * (self.ymax - self.ymin),
                "theta": self.thetamin
                + torch.rand((self.boundary_right_samples, 1))
                * (self.thetamax - self.thetamin),
                # "theta": torch.pi/2 + torch.rand((self.boundary_right_samples, 1)) * torch.pi,
            },
            "boundary_bottom": {
                "x": self.xmin
                + torch.rand((self.boundary_bottom_samples, 1))
                * (self.xmax - self.xmin),
                "y": self.ymin * torch.ones((self.boundary_bottom_samples, 1)),
                "theta": self.thetamin
                + torch.rand((self.boundary_bottom_samples, 1))
                * (self.thetamax - self.thetamin),
                # "theta": torch.rand((self.boundary_bottom_samples, 1)) * torch.pi,
            },
            "boundary_top": {
                "x": self.xmin
                + torch.rand((self.boundary_top_samples, 1))
                * (self.xmax - self.xmin),
                "y": self.ymax * torch.ones((self.boundary_top_samples, 1)),
                "theta": self.thetamin
                + torch.rand((self.boundary_top_samples, 1))
                * (self.thetamax - self.thetamin),
                # "theta": -torch.rand((self.boundary_top_samples, 1)) * torch.pi,
            },
        }
        return tensor_dict


def Data_Loader(config):
    dataset = Dataset(config)
    dataloader = data.DataLoader(dataset, batch_size=None, num_workers=0)
    return iter(dataloader)
