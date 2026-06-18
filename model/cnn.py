import logging
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
    ):
        super().__init__()

        # Using GroupNorm instead of BatchNorm because it's compatible with DP
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv1_channels, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(4, conv1_channels), # 4 groups
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(4, conv2_channels), # 4 groups
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        flattened_dim = self._get_flattened_dim(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
        )

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

        logging.info(
            "LeNet-5 with GroupNorm initialized | "
            f"params={sum(p.numel() for p in self.parameters())}"
        )

    def _get_flattened_dim(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
    ) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            out = self.features(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x