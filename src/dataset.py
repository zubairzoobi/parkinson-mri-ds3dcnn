from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# The class order must remain unchanged when using trained checkpoints.
CLASS_NAMES = ["NC", "PD", "Prodromal"]

CLASS_TO_INDEX = {
    class_name: class_index
    for class_index, class_name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    class_index: class_name
    for class_name, class_index in CLASS_TO_INDEX.items()
}


def preprocess_volume(
    image_path: str | Path,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess one three-dimensional NIfTI MRI volume.

    Processing steps:
    1. Load the NIfTI image as float32.
    2. Verify that the image is three-dimensional.
    3. Check for NaN or infinite values.
    4. Clip intensities between the 1st and 99th percentiles.
    5. Apply subject-wise z-score normalization.
    6. Add a channel dimension.

    Parameters
    ----------
    image_path:
        Path to a .nii or .nii.gz MRI file.

    Returns
    -------
    tensor:
        Preprocessed MRI tensor with shape:
        [1, depth, height, width]

    affine:
        Original NIfTI affine matrix.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"NIfTI image was not found: {image_path}"
        )

    nifti_image = nib.load(str(image_path))

    image = nifti_image.get_fdata(
        dtype=np.float32
    )

    if image.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional MRI volume, "
            f"but received shape {image.shape}: {image_path}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            f"The MRI volume contains NaN or infinite values: "
            f"{image_path}"
        )

    # Reduce the influence of extreme intensity values.
    percentile_1, percentile_99 = np.percentile(
        image,
        (1, 99),
    )

    image = np.clip(
        image,
        percentile_1,
        percentile_99,
    )

    mean_intensity = float(image.mean())
    standard_deviation = float(image.std())

    if standard_deviation < 1e-8:
        raise ValueError(
            f"The MRI volume has near-zero intensity variance: "
            f"{image_path}"
        )

    # Subject-wise z-score normalization.
    image = (
        image - mean_intensity
    ) / (
        standard_deviation + 1e-8
    )

    # Convert the NumPy array to a PyTorch tensor.
    # The first dimension represents the single MRI channel.
    tensor = torch.from_numpy(
        image.astype(np.float32)
    ).unsqueeze(0)

    return tensor, nifti_image.affine


class NiftiDirectoryDataset(Dataset):
    """
    PyTorch dataset for loading MRI volumes from class folders.

    Expected directory structure:

    dataset/
        NC/
        PD/
        Prodromal/

    Each folder may contain .nii or .nii.gz files.
    """

    def __init__(
        self,
        directory: str | Path,
        classes: list[str] | None = None,
        expected_shape: tuple[int, int, int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        directory:
            Root directory containing the class folders.

        classes:
            Ordered list of class-folder names.

        expected_shape:
            Optional expected spatial shape of every MRI volume.
            For the current experiment, this is:
            (193, 229, 193)
        """

        self.directory = Path(directory)

        self.classes = (
            classes
            if classes is not None
            else CLASS_NAMES
        )

        self.expected_shape = expected_shape

        self.files: list[Path] = []
        self.labels: list[int] = []

        if not self.directory.exists():
            raise FileNotFoundError(
                f"Dataset directory was not found: "
                f"{self.directory}"
            )

        for label_index, class_name in enumerate(
            self.classes
        ):
            class_directory = (
                self.directory / class_name
            )

            if not class_directory.is_dir():
                raise FileNotFoundError(
                    f"Class folder was not found: "
                    f"{class_directory}"
                )

            class_files = sorted(
                file_path
                for file_path in class_directory.iterdir()
                if (
                    file_path.name.endswith(".nii")
                    or file_path.name.endswith(".nii.gz")
                )
            )

            self.files.extend(class_files)

            self.labels.extend(
                [label_index] * len(class_files)
            )

        if not self.files:
            raise ValueError(
                f"No NIfTI files were found under: "
                f"{self.directory}"
            )

        self.labels_array = np.asarray(
            self.labels,
            dtype=int,
        )

        self._print_dataset_summary()

    def _print_dataset_summary(self) -> None:
        """Print the number of MRI volumes in each class."""

        print("\nDataset summary:")

        for class_index, class_name in enumerate(
            self.classes
        ):
            class_count = int(
                np.sum(
                    self.labels_array == class_index
                )
            )

            print(
                f"{class_name}: "
                f"{class_count} subjects"
            )

        print(
            f"Total: {len(self.files)} "
            f"NIfTI files / subjects"
        )

    def __len__(self) -> int:
        """Return the number of MRI volumes."""

        return len(self.files)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Load and return one preprocessed MRI and its label.
        """

        image_path = self.files[index]

        image_tensor, _ = preprocess_volume(
            image_path
        )

        if self.expected_shape is not None:
            actual_shape = tuple(
                image_tensor.shape[1:]
            )

            if actual_shape != self.expected_shape:
                raise ValueError(
                    f"Unexpected MRI shape {actual_shape}; "
                    f"expected {self.expected_shape}: "
                    f"{image_path}"
                )

        label = int(
            self.labels[index]
        )

        return image_tensor, label
