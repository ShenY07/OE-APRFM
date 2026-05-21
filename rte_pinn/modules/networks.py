import torch
import torch.nn as nn
import os


class AdaptiveResidualBlock(nn.Module):
    def __init__(
        self,
        units,
        activation,
        device,
        adaptive_rate=0.9,
        name="residual_block",
        **kwargs,
    ):
        super(AdaptiveResidualBlock, self).__init__()
        self._units = units
        self._layers = nn.ModuleList([
            nn.Linear(units[i], units[i]) for i in range(len(units))
        ])
        self._activation = activation
        self._device = device
        self._beta = nn.Parameter(torch.tensor(adaptive_rate).to(self._device))

    def forward(self, inputs):
        beta = torch.clamp(self._beta, min=0.05, max=1.0)
        residual = inputs
        for _, h_i in enumerate(self._layers):
            inputs = self._activation(h_i(inputs))
        residual = beta * residual + (1.0 - beta) * inputs
        return residual


class ResNet_1d(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, device):
        super(ResNet_1d, self).__init__()

        self._layers = hidden_sizes
        self._input_layer = nn.Linear(input_size, hidden_sizes[0])
        self._device = device
        self._residual_blocks = nn.ModuleList()
        self._residual_blocks.append(self._input_layer)

        for i in range(1, len(self._layers) - 1, 2):
            self._residual_blocks.append(
                AdaptiveResidualBlock(
                    units=self._layers[i: i + 2],
                    activation=self.activation,
                    device=self._device,
                )
            )

        self._output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self._residual_blocks.append(self._output_layer)

    def forward(self, inputs):
        output = inputs
        for i in range(len(self._residual_blocks)):
            output = self._residual_blocks[i](output)
        output = self.positive(output)  # ensure positivity
        return output

    def activation(self, o):
        return torch.tanh(o)

    def positive(self, o):
        return torch.log(1.0 + torch.exp(o))


class ResNet_2d(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, freq, device):
        super(ResNet_2d, self).__init__()

        self._layers = hidden_sizes
        self._input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.freq = freq
        self._device = device
        self._residual_blocks = nn.ModuleList()
        self._residual_blocks.append(self._input_layer)

        for i in range(1, len(self._layers) - 1, 2):
            self._residual_blocks.append(
                AdaptiveResidualBlock(
                    units=self._layers[i: i + 2],
                    activation=self.activation,
                    device=self._device,
                )
            )

        self._output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self._residual_blocks.append(self._output_layer)

    def forward(self, inputs):
        output = self.transform(inputs)  # apply periodic transform
        for i in range(len(self._residual_blocks)):
            output = self._residual_blocks[i](output)
        return output

    def activation(self, o):
        return torch.tanh(o)

    # periodic in x, y
    def transform(self, xytheta):
        x, y, theta = torch.split(xytheta, [1, 1, 1], dim=-1)
        xi, eta = torch.cos(theta), torch.sin(theta)
        xytheta = torch.cat([x, y, xi, eta], dim=-1)
        return xytheta


def Xavier_initi(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def save_param(net, path):
    torch.save(net.state_dict(), path)


def load_param(net, path):
    if os.path.exists(path):
        # Load to the same device as the network
        net.load_state_dict(
            torch.load(path, map_location=next(net.parameters()).device)
        )
    else:
        print("File does not exist.")
