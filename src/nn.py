import torch
from torch.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    LocalResponseNorm,
    MaxPool2d,
    Module,
    ReLU,
)


class DBlock(Module):
    """Encapsulated AlexNet block.

    Parameters
    ----------
    in_channels : int
        Number of expected features in the input.
    out_channels : int
        Number of output features.
    kernel_size : tuple[int, int] | int
        Kernel size to use for the convolution.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int] | int
    ) -> None:
        super(DBlock, self).__init__()

        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )

        self.relu = ReLU(inplace=True)
        self.loc_norm = LocalResponseNorm(size=out_channels)
        self.max_pool = MaxPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input.

        Parameters
        ----------
        x : torch.Tensor
            Input into the layer.

        Returns
        -------
        torch.Tensor
            Features extracted from the DBlock.
        """
        x = self.relu(self.conv(x))
        x = self.loc_norm(x)
        return self.max_pool(x)


class Head(Module):
    """Embedding extractor.

    Parameters
    ----------
    in_features : int
        Number of expected features in the input.
    out_features : int
        Number of output features.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super(Head, self).__init__()

        self.linear_1 = Linear(in_features=in_features, out_features=256)
        self.relu_1 = ReLU()
        self.dropout = Dropout(p=0.5)
        self.linear_2 = Linear(in_features=256, out_features=out_features)
        self.relu_2 = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input.

        Parameters
        ----------
        x : torch.Tensor
            Input into the layer.

        Returns
        -------
        torch.Tensor
            Features extracted from the DBlock.
        """
        features = self.relu_1(self.linear_1(x))
        features = self.dropout(features)
        features = self.relu_2(self.linear_2(features))
        return features


class SiameeseNN(Module):
    """Siameese network with AlexNet backbone."""

    def __init__(self) -> None:
        super(SiameeseNN, self).__init__()
        self.block_1 = DBlock(in_channels=3, out_channels=96, kernel_size=11)
        self.block_2 = DBlock(in_channels=96, out_channels=256, kernel_size=5)
        self.block_3 = DBlock(in_channels=256, out_channels=384, kernel_size=3)
        self.block_4 = DBlock(in_channels=384, out_channels=256, kernel_size=3)
        self.flatten = Flatten()
        self.head = Head(in_features=65536, out_features=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate forward the input.

        Parameters
        ----------
        x : torch.Tensor
            Input into the layer.

        Returns
        -------
        torch.Tensor
            Features extracted from the DBlock.
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.flatten(x)
        return self.head(x)
