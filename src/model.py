from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DepthwiseSeparableConv3D(nn.Module):
    """
    Depthwise separable 3D convolution block.

    This block contains:

    1. Depthwise 3D convolution
    2. Pointwise 1 x 1 x 1 convolution
    3. Batch normalization
    4. ReLU activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()

        # Depthwise convolution applies one spatial filter
        # independently to every input channel.
        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        # Pointwise convolution combines information
        # across different channels.
        self.point_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm3d(
            out_channels
        )

        self.act = nn.ReLU(
            inplace=True
        )

        # Weight initialization.
        nn.init.kaiming_normal_(
            self.depth_conv.weight,
            nonlinearity="relu",
        )

        nn.init.kaiming_normal_(
            self.point_conv.weight,
            nonlinearity="relu",
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass through the convolution block.
        """

        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DS3DCNN(nn.Module):
    """
    Depthwise Separable 3D CNN for MRI classification.

    Default input MRI shape:

        193 x 229 x 193

    Default class order:

        NC = 0
        PD = 1
        Prodromal = 2
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_shape: Sequence[int] = (
            193,
            229,
            193,
        ),
        dropout_3d: float = 0.10,
        dropout_fc: float = 0.20,
    ) -> None:
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError(
                "input_shape must contain exactly "
                "three spatial dimensions."
            )

        self.input_shape = tuple(
            int(dimension)
            for dimension in input_shape
        )

        if any(
            dimension < 8
            for dimension in self.input_shape
        ):
            raise ValueError(
                "Each MRI spatial dimension must be "
                "at least 8 voxels."
            )

        # Block 1:
        # Input channels: 1
        # Output channels: 32
        self.block1 = DepthwiseSeparableConv3D(
            in_channels=1,
            out_channels=32,
        )

        self.pool1 = nn.MaxPool3d(
            kernel_size=2
        )

        self.drop1 = nn.Dropout3d(
            p=dropout_3d
        )

        # Block 2:
        # Input channels: 32
        # Output channels: 64
        self.block2 = DepthwiseSeparableConv3D(
            in_channels=32,
            out_channels=64,
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=2
        )

        self.drop2 = nn.Dropout3d(
            p=dropout_3d
        )

        # Block 3:
        # Input channels: 64
        # Output channels: 128
        self.block3 = DepthwiseSeparableConv3D(
            in_channels=64,
            out_channels=128,
        )

        self.pool3 = nn.MaxPool3d(
            kernel_size=2
        )

        self.drop_fc = nn.Dropout(
            p=dropout_fc
        )

        # Three max-pooling operations divide each
        # spatial dimension by approximately 8.
        #
        # Default input:
        # 193 x 229 x 193
        #
        # After pooling:
        # 24 x 28 x 24
        pooled_shape = tuple(
            dimension // 8
            for dimension in self.input_shape
        )

        flattened_features = (
            128
            * pooled_shape[0]
            * pooled_shape[1]
            * pooled_shape[2]
        )

        self.fc = nn.Linear(
            in_features=flattened_features,
            out_features=num_classes,
        )

        # Initialize the fully connected layer.
        nn.init.xavier_normal_(
            self.fc.weight
        )

        nn.init.constant_(
            self.fc.bias,
            0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the DS-3DCNN forward pass.

        Parameters
        ----------
        x:
            Input MRI tensor with shape:

            batch_size x 1 x depth x height x width

        Returns
        -------
        torch.Tensor:
            Raw classification logits with shape:

            batch_size x num_classes
        """

        if x.ndim != 5:
            raise ValueError(
                "Expected a five-dimensional input tensor "
                "with shape [batch, channel, depth, height, width], "
                f"but received shape {tuple(x.shape)}."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "The model expects one MRI input channel, "
                f"but received {x.shape[1]} channels."
            )

        # First convolution block.
        x = self.block1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Second convolution block.
        x = self.block2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Third convolution block.
        x = self.block3(x)
        x = self.pool3(x)

        # Flatten all convolution features.
        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.drop_fc(x)

        # Raw logits are returned.
        # CrossEntropyLoss applies softmax internally.
        x = self.fc(x)

        return x
