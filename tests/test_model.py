from __future__ import annotations

import pytest
import torch

from src.model import (
    DS3DCNN,
    DepthwiseSeparableConv3D,
)


def test_depthwise_separable_block_output_shape() -> None:
    """
    Check that a depthwise separable convolution block
    produces the expected number of output channels.
    """

    block = DepthwiseSeparableConv3D(
        in_channels=1,
        out_channels=32,
    )

    block.eval()

    input_tensor = torch.randn(
        2,
        1,
        16,
        16,
        16,
    )

    with torch.no_grad():
        output_tensor = block(
            input_tensor
        )

    assert output_tensor.shape == (
        2,
        32,
        16,
        16,
        16,
    )


def test_model_output_shape() -> None:
    """
    Check that DS-3DCNN returns one logit for each
    diagnostic class.

    A smaller input shape is used to keep the test fast
    and memory efficient.
    """

    model = DS3DCNN(
        num_classes=3,
        input_shape=(
            32,
            32,
            32,
        ),
        dropout_3d=0.10,
        dropout_fc=0.20,
    )

    model.eval()

    input_tensor = torch.randn(
        2,
        1,
        32,
        32,
        32,
    )

    with torch.no_grad():
        logits = model(
            input_tensor
        )

    assert logits.shape == (
        2,
        3,
    )

    assert torch.isfinite(
        logits
    ).all()


def test_checkpoint_compatible_layer_names() -> None:
    """
    Check that layer names remain compatible with
    the original trained DS-3DCNN checkpoints.
    """

    model = DS3DCNN(
        num_classes=3,
        input_shape=(
            32,
            32,
            32,
        ),
    )

    assert hasattr(
        model.block1,
        "depth_conv",
    )

    assert hasattr(
        model.block1,
        "point_conv",
    )

    assert hasattr(
        model.block1,
        "bn",
    )

    assert hasattr(
        model.block1,
        "act",
    )

    assert hasattr(
        model,
        "drop1",
    )

    assert hasattr(
        model,
        "drop2",
    )

    assert hasattr(
        model,
        "drop_fc",
    )

    assert hasattr(
        model,
        "fc",
    )


def test_invalid_number_of_input_dimensions() -> None:
    """
    Check that the model rejects an input tensor
    that does not contain five dimensions.
    """

    model = DS3DCNN(
        num_classes=3,
        input_shape=(
            32,
            32,
            32,
        ),
    )

    invalid_input = torch.randn(
        1,
        32,
        32,
        32,
    )

    with pytest.raises(
        ValueError,
        match="five-dimensional",
    ):
        model(
            invalid_input
        )


def test_invalid_number_of_channels() -> None:
    """
    Check that the model rejects MRI tensors with
    more than one input channel.
    """

    model = DS3DCNN(
        num_classes=3,
        input_shape=(
            32,
            32,
            32,
        ),
    )

    invalid_input = torch.randn(
        1,
        2,
        32,
        32,
        32,
    )

    with pytest.raises(
        ValueError,
        match="one MRI input channel",
    ):
        model(
            invalid_input
        )


def test_invalid_input_shape_configuration() -> None:
    """
    Check that the model rejects a configuration
    that does not contain three spatial dimensions.
    """

    with pytest.raises(
        ValueError,
        match="three spatial dimensions",
    ):
        DS3DCNN(
            input_shape=(
                32,
                32,
            ),
        )


def test_spatial_dimensions_smaller_than_eight() -> None:
    """
    Check that dimensions too small for three pooling
    operations are rejected.
    """

    with pytest.raises(
        ValueError,
        match="at least 8 voxels",
    ):
        DS3DCNN(
            input_shape=(
                4,
                32,
                32,
            ),
        )
