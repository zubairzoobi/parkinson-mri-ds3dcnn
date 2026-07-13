from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Use a non-interactive backend so plots can also be
# generated on a remote server without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    preprocess_volume,
)
from src.train import create_model
from src.utils import (
    load_config,
    load_model_checkpoint,
    select_device,
    set_seed,
)


class GradCAM3D:
    """
    Generate a three-dimensional Grad-CAM heatmap.

    The class captures activations and gradients from a
    selected convolutional layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Trained DS-3DCNN model.

        target_layer:
            Convolutional block used for Grad-CAM.
        """

        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """
        Save layer activations and register a gradient hook.
        """

        del module
        del inputs

        self.activations = output.detach()

        output.register_hook(
            self._save_gradients
        )

    def _save_gradients(
        self,
        gradients: torch.Tensor,
    ) -> None:
        """Save gradients produced during backpropagation."""

        self.gradients = gradients.detach()

    def generate(
        self,
        image: torch.Tensor,
        target_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a Grad-CAM heatmap for one target class.

        Parameters
        ----------
        image:
            Input MRI batch with shape:

            [1, 1, depth, height, width]

        target_index:
            Integer index of the target diagnostic class.

        Returns
        -------
        heatmap:
            Normalized three-dimensional Grad-CAM heatmap.

        logits:
            Raw model output logits.
        """

        if image.ndim != 5:
            raise ValueError(
                "Grad-CAM expects a five-dimensional tensor "
                "with shape [batch, channel, depth, height, width]."
            )

        if image.shape[0] != 1:
            raise ValueError(
                "Grad-CAM currently supports one MRI "
                "volume at a time."
            )

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            image
        )

        if target_index < 0 or target_index >= logits.shape[1]:
            raise ValueError(
                f"Invalid target class index: {target_index}"
            )

        target_score = logits[
            :,
            target_index,
        ].sum()

        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM did not capture layer activations."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM did not capture layer gradients."
            )

        # Global average pooling of gradients over the
        # three spatial dimensions.
        weights = self.gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        # Weighted combination of feature maps.
        heatmap = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        heatmap = F.relu(
            heatmap
        )

        # Resize the Grad-CAM map to the original
        # MRI spatial dimensions.
        heatmap = F.interpolate(
            heatmap,
            size=image.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        # Normalize the heatmap to the range 0 to 1.
        minimum = heatmap.amin(
            dim=(2, 3, 4),
            keepdim=True,
        )

        maximum = heatmap.amax(
            dim=(2, 3, 4),
            keepdim=True,
        )

        heatmap = (
            heatmap - minimum
        ) / (
            maximum - minimum + 1e-8
        )

        return heatmap, logits.detach()

    def close(self) -> None:
        """Remove the registered forward hook."""

        self.forward_handle.remove()


def save_center_slice_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    target_class: str,
    predicted_class: str,
    confidence: float,
) -> None:
    """
    Save an MRI and Grad-CAM center-slice overlay.

    Parameters
    ----------
    image:
        Preprocessed three-dimensional MRI volume.

    heatmap:
        Three-dimensional Grad-CAM heatmap.

    output_path:
        Destination PNG file.

    target_class:
        Class used for Grad-CAM generation.

    predicted_class:
        Class predicted by the model.

    confidence:
        Softmax confidence of the predicted class.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    slice_index = (
        image.shape[2] // 2
    )

    background_slice = image[
        :,
        :,
        slice_index,
    ]

    activation_slice = heatmap[
        :,
        :,
        slice_index,
    ]

    figure, axis = plt.subplots(
        figsize=(8, 8)
    )

    axis.imshow(
        np.rot90(
            background_slice
        ),
        cmap="gray",
    )

    overlay = axis.imshow(
        np.rot90(
            activation_slice
        ),
        cmap="jet",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )

    figure.colorbar(
        overlay,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Grad-CAM activation",
    )

    axis.set_title(
        f"Target: {target_class} | "
        f"Prediction: {predicted_class} "
        f"({confidence:.2%})"
    )

    axis.axis(
        "off"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main(
    config_path: str,
    checkpoint_path: str,
    image_path: str,
    target_class: str,
    output_prefix: str,
) -> None:
    """
    Load a trained model and generate a 3D Grad-CAM map.
    """

    if target_class not in CLASS_TO_INDEX:
        raise ValueError(
            f"Unknown target class '{target_class}'. "
            f"Valid classes are: {CLASS_NAMES}"
        )

    config = load_config(
        config_path
    )

    set_seed(
        int(config["seed"])
    )

    configured_classes = list(
        config["data"]["classes"]
    )

    if configured_classes != CLASS_NAMES:
        raise ValueError(
            "The configured class order must be exactly: "
            "NC, PD, Prodromal."
        )

    device = select_device()

    model = create_model(
        config
    ).to(device)

    load_model_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.eval()

    image_tensor, affine = preprocess_volume(
        image_path
    )

    expected_shape = tuple(
        int(value)
        for value in config[
            "data"
        ][
            "input_shape"
        ]
    )

    actual_shape = tuple(
        image_tensor.shape[1:]
    )

    if actual_shape != expected_shape:
        raise ValueError(
            f"Image shape {actual_shape} does not match "
            f"configured input shape {expected_shape}."
        )

    input_batch = image_tensor.unsqueeze(
        0
    ).to(device)

    # The third convolution block is used because it
    # contains higher-level spatial features while still
    # preserving three-dimensional location information.
    gradcam = GradCAM3D(
        model=model,
        target_layer=model.block3,
    )

    try:
        heatmap_tensor, logits = gradcam.generate(
            image=input_batch,
            target_index=CLASS_TO_INDEX[
                target_class
            ],
        )

    finally:
        gradcam.close()

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    predicted_index = int(
        probabilities.argmax(
            dim=1
        ).item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    prediction_confidence = float(
        probabilities[
            0,
            predicted_index,
        ].item()
    )

    print(
        "\nModel prediction"
    )

    print(
        "=" * 50
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        class_probability = float(
            probabilities[
                0,
                class_index,
            ].item()
        )

        print(
            f"{class_name:<12}: "
            f"{class_probability:.4f}"
        )

    print(
        f"\nPredicted class: "
        f"{predicted_class}"
    )

    print(
        f"Confidence: "
        f"{prediction_confidence:.4f}"
    )

    print(
        f"Grad-CAM target class: "
        f"{target_class}"
    )

    heatmap = (
        heatmap_tensor
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    image_array = (
        image_tensor
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    output_prefix_path = Path(
        output_prefix
    )

    output_prefix_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nifti_output_path = Path(
        f"{output_prefix_path}.nii.gz"
    )

    png_output_path = Path(
        f"{output_prefix_path}.png"
    )

    gradcam_nifti = nib.Nifti1Image(
        heatmap,
        affine,
    )

    nib.save(
        gradcam_nifti,
        str(nifti_output_path),
    )

    save_center_slice_overlay(
        image=image_array,
        heatmap=heatmap,
        output_path=png_output_path,
        target_class=target_class,
        predicted_class=predicted_class,
        confidence=prediction_confidence,
    )

    print(
        f"\n3D Grad-CAM saved to: "
        f"{nifti_output_path.resolve()}"
    )

    print(
        f"PNG overlay saved to: "
        f"{png_output_path.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a three-dimensional Grad-CAM "
            "for a trained Parkinson MRI DS-3DCNN."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help=(
            "Path to the YAML configuration file."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to the trained .pth checkpoint."
        ),
    )

    parser.add_argument(
        "--image",
        required=True,
        help=(
            "Path to one anonymized .nii or .nii.gz MRI."
        ),
    )

    parser.add_argument(
        "--target-class",
        required=True,
        choices=CLASS_NAMES,
        help=(
            "Class for which Grad-CAM will be generated."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        required=True,
        help=(
            "Output path without a file extension."
        ),
    )

    arguments = parser.parse_args()

    main(
        config_path=arguments.config,
        checkpoint_path=arguments.checkpoint,
        image_path=arguments.image,
        target_class=arguments.target_class,
        output_prefix=arguments.output_prefix,
    )
