import torch
from torch.utils import data


class Dataset(data.IterableDataset):
    def __init__(self, config):
        self.domain = config.domain
        self.dataset = config.model.dataset
        self.xmin, self.xmax = self.domain.x
        self.ymin, self.ymax = self.domain.y

        # 洞的范围 (hxmin, hxmax, hymin, hymax)
        self.hole = config.domain.hole
        self.hxmin, self.hxmax = self.hole.x
        self.hymin, self.hymax = self.hole.y

        self.thetamin, self.thetamax = self.domain.theta

        self.interior_samples = self.dataset.interior_samples
        self.boundary_left_samples = self.dataset.boundary_left_samples
        self.boundary_right_samples = self.dataset.boundary_right_samples
        self.boundary_bottom_samples = self.dataset.boundary_bottom_samples
        self.boundary_top_samples = self.dataset.boundary_top_samples
        self.hole_left_samples = self.dataset.hole_left_samples
        self.hole_right_samples = self.dataset.hole_right_samples
        self.hole_bottom_samples = self.dataset.hole_bottom_samples
        self.hole_top_samples = self.dataset.hole_top_samples

    def __iter__(self):
        return self

    def __next__(self):
        # -------------------
        # Interior points (avoid hole)
        # -------------------
        num_points = self.interior_samples

        # 每块点数（简单均分）
        n_side = num_points // 4
        n_remain = num_points - n_side*3  # 剩余给最后一块

        # 左侧块
        x_left = self.xmin + torch.rand((n_side, 1)) * (self.hxmin - self.xmin)
        y_left = self.ymin + torch.rand((n_side, 1)) * (self.ymax - self.ymin)

        # 右侧块
        x_right = self.hxmax + \
            torch.rand((n_side, 1)) * (self.xmax - self.hxmax)
        y_right = self.ymin + torch.rand((n_side, 1)) * (self.ymax - self.ymin)

        # 下方块
        x_bottom = self.hxmin + \
            torch.rand((n_side, 1)) * (self.hxmax - self.hxmin)
        y_bottom = self.ymin + \
            torch.rand((n_side, 1)) * (self.hymin - self.ymin)

        # 上方块
        x_top = self.hxmin + \
            torch.rand((n_remain, 1)) * (self.hxmax - self.hxmin)
        y_top = self.hymax + \
            torch.rand((n_remain, 1)) * (self.ymax - self.hymax)

        # 拼接
        interior_x = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
        interior_y = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)

        # theta
        interior_theta = self.thetamin + \
            torch.rand((num_points, 1)) * (self.thetamax - self.thetamin)
        # -------------------
        # External boundaries
        # -------------------
        boundary_left = {
            "x": self.xmin * torch.ones((self.boundary_left_samples, 1)),
            "y": self.ymin + torch.rand((self.boundary_left_samples, 1)) * (self.ymax - self.ymin),
            "theta": self.thetamin + torch.rand((self.boundary_left_samples, 1)) * (self.thetamax - self.thetamin),
        }
        boundary_right = {
            "x": self.xmax * torch.ones((self.boundary_right_samples, 1)),
            "y": self.ymin + torch.rand((self.boundary_right_samples, 1)) * (self.ymax - self.ymin),
            "theta": self.thetamin + torch.rand((self.boundary_right_samples, 1)) * (self.thetamax - self.thetamin),
        }
        boundary_bottom = {
            "x": self.xmin + torch.rand((self.boundary_bottom_samples, 1)) * (self.xmax - self.xmin),
            "y": self.ymin * torch.ones((self.boundary_bottom_samples, 1)),
            "theta": self.thetamin + torch.rand((self.boundary_bottom_samples, 1)) * (self.thetamax - self.thetamin),
        }
        boundary_top = {
            "x": self.xmin + torch.rand((self.boundary_top_samples, 1)) * (self.xmax - self.xmin),
            "y": self.ymax * torch.ones((self.boundary_top_samples, 1)),
            "theta": self.thetamin + torch.rand((self.boundary_top_samples, 1)) * (self.thetamax - self.thetamin),
        }

        # -------------------
        # Hole boundaries
        # -------------------
        hole_left = {
            "x": self.hxmax * torch.ones((self.hole_left_samples, 1)),
            "y": self.hymin + torch.rand((self.hole_left_samples, 1)) * (self.hymax - self.hymin),
            "theta": self.thetamin + torch.rand((self.hole_left_samples, 1)) * (self.thetamax - self.thetamin),
        }
        hole_right = {
            "x": self.hxmin * torch.ones((self.hole_right_samples, 1)),
            "y": self.hymin + torch.rand((self.hole_right_samples, 1)) * (self.hymax - self.hymin),
            "theta": self.thetamin + torch.rand((self.hole_right_samples, 1)) * (self.thetamax - self.thetamin),
        }
        hole_bottom = {
            "x": self.hxmin + torch.rand((self.hole_bottom_samples, 1)) * (self.hxmax - self.hxmin),
            "y": self.hymax * torch.ones((self.hole_bottom_samples, 1)),
            "theta": self.thetamin + torch.rand((self.hole_bottom_samples, 1)) * (self.thetamax - self.thetamin),
        }
        hole_top = {
            "x": self.hxmin + torch.rand((self.hole_top_samples, 1)) * (self.hxmax - self.hxmin),
            "y": self.hymin * torch.ones((self.hole_top_samples, 1)),
            "theta": self.thetamin + torch.rand((self.hole_top_samples, 1)) * (self.thetamax - self.thetamin),
        }

        tensor_dict = {
            "interior": {"x": interior_x, "y": interior_y, "theta": interior_theta},
            "boundary_left": boundary_left,
            "boundary_right": boundary_right,
            "boundary_bottom": boundary_bottom,
            "boundary_top": boundary_top,
            "hole_left": hole_left,
            "hole_right": hole_right,
            "hole_bottom": hole_bottom,
            "hole_top": hole_top,
        }

        return tensor_dict


def Data_Loader(config):
    dataset = Dataset(config)
    dataloader = data.DataLoader(dataset, batch_size=None, num_workers=0)
    return iter(dataloader)
