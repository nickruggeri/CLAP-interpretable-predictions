"""Module for the neural network architectures utilized in CLAP. """
import torch
import torch.nn as nn


class CLAPEncoderBackbone(nn.Module):
    def __init__(
        self, n_channels: int, in_dim: int, intermediate_size: int = 256
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.in_dim = in_dim
        self.intermediate_size = intermediate_size

        # https://arxiv.org/pdf/1912.00155.pdf
        self.fc = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Flatten(),
            nn.Linear(1024, self.intermediate_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.fc(x)
        return encoded


class CLAPDecoder(nn.Module):
    def __init__(self, n_channels: int, z_core_dim: int, z_style_dim: int) -> None:
        super().__init__()
        self.z_dim = z_core_dim + z_style_dim
        self.n_channels = n_channels

        # https://arxiv.org/pdf/1912.00155.pdf
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            View(-1, 64, 4, 4),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, self.n_channels, kernel_size=4, stride=2, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.fc(z)
        return decoded

    def get_first_linear_layer(self) -> torch.Tensor:
        return next(self.fc.parameters())


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def get_weights(self) -> torch.Tensor:
        return next(self.fc.parameters())


class View(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.dim)
