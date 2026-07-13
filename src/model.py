from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DepthwiseSeparableConv3D(nn.Module):
    """
    Depthwise separable three-dimensional convolution.

    The operation consists of:

    1. A depthwise 3D convolution that applies one spatial
       filter independently to each input channel.

    2. A pointwise 1 x 1 x 1 convolution that combines
       information across channels.

    3. Batch normalization.

    4. ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()

        self.depth_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.point_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm3d(
            out_channels
        )

        self.activation = nn.ReLU(
            inplace=True
        )

        # Kaiming initialization is suitable for ReLU networks.
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
        Apply depthwise convolution, pointwise convolution,
        batch normalization, and ReLU activation.
        """

        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x


class DS3DCNN(nn.Module):
    """
    Depthwise Separable 3D CNN for three-class MRI classification.

    Default class order:

    NC = 0
    PD = 1
    Prodromal = 2

    Default input shape:

    193 x 229 x 193
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
            int(value)
            for value in input_shape
        )

        # First depthwise separable convolution block.
        self.block1 = DepthwiseSeparableConv3D(
            in_channels=1,
            out_channels=32,
        )

        self.pool1 = nn.MaxPool3d(
            kernel_size=2
        )

        self.dropout1 = nn.Dropout3d(
            dropout_3d
        )

        # Second depthwise separable convolution block.
        self.block2 = DepthwiseSeparableConv3D(
            in_channels=32,
            out_channels=64,
        )

        self.pool2 = nn.MaxPool3d(
            kernel_size=2
        )

        self.dropout2 = nn.Dropout3d(
            dropout_3d
        )

        # Third depthwise separable convolution block.
        self.block3 = DepthwiseSeparableConv3D(
            in_channels=64,
            out_channels=128,
        )

        self.pool3 = nn.MaxPool3d(
            kernel_size=2
        )

        self.dropout_classifier = nn.Dropout(
            dropout_fc
        )

        # Three max-pooling operations reduce each spatial
        # dimension by a factor of 8.
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

        self.classifier = nn.Linear(
            in_features=flattened_features,
            out_features=num_classes,
        )

        nn.init.xavier_normal_(
            self.classifier.weight
        )

        nn.init.constant_(
            self.classifier.bias,
            0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the forward pass.

        Parameters
        ----------
        x:
            MRI tensor with shape:

            batch_size x 1 x depth x height x width

        Returns
        -------
        logits:
            Raw output logits for the three classes.
        """

        x = self.block1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.block2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.dropout_classifier(x)

        logits = self.classifier(x)

        return logits
