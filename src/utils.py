from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """
    Load experiment settings from a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict:
        Parsed configuration dictionary.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file was not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration file must contain "
            "a YAML dictionary."
        )

    return config


def set_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Integer random seed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducible CUDA behaviour.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """
    Select CUDA GPU when available; otherwise use CPU.

    Returns
    -------
    torch.device:
        Selected PyTorch computation device.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(
            torch.cuda.current_device()
        )

        print(
            f"Using CUDA GPU: {gpu_name}"
        )

    else:
        device = torch.device("cpu")

        print(
            "CUDA GPU was not found. Using CPU."
        )

    return device


def ensure_directory(
    directory: str | Path,
) -> Path:
    """
    Create a directory when it does not already exist.

    Parameters
    ----------
    directory:
        Directory path.

    Returns
    -------
    Path:
        Created or existing directory path.
    """

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save a dictionary as a formatted JSON file.

    NumPy arrays and NumPy numeric values are converted
    into standard Python objects before saving.

    Parameters
    ----------
    data:
        Dictionary to save.

    output_path:
        Destination JSON file.
    """

    output_path = Path(output_path)

    ensure_directory(
        output_path.parent
    )

    def convert_value(
        value: Any,
    ) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(
            value,
            (
                np.integer,
                np.floating,
            ),
        ):
            return value.item()

        if isinstance(value, Path):
            return str(value)

        raise TypeError(
            f"Object of type {type(value).__name__} "
            "is not JSON serializable."
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            default=convert_value,
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_f1: float,
    validation_loss: float,
    output_path: str | Path,
) -> None:
    """
    Save a model checkpoint.

    Parameters
    ----------
    model:
        PyTorch model.

    optimizer:
        Training optimizer.

    epoch:
        Epoch number.

    validation_f1:
        Validation macro F1-score.

    validation_loss:
        Validation loss.

    output_path:
        Destination checkpoint path.
    """

    output_path = Path(output_path)

    ensure_directory(
        output_path.parent
    )

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_f1": float(validation_f1),
        "validation_loss": float(validation_loss),
    }

    torch.save(
        checkpoint,
        output_path,
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load model weights from a saved checkpoint.

    The function supports:

    1. A checkpoint dictionary containing model_state_dict.
    2. A plain PyTorch state dictionary.

    Parameters
    ----------
    model:
        Model instance receiving the weights.

    checkpoint_path:
        Path to the saved .pth checkpoint.

    device:
        Device used to load the checkpoint.

    Returns
    -------
    dict:
        Loaded checkpoint dictionary.
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        return checkpoint

    if isinstance(checkpoint, dict):
        model.load_state_dict(
            checkpoint
        )

        return {
            "model_state_dict": checkpoint
        }

    raise ValueError(
        "Unsupported checkpoint format."
    )


def plot_confusion_matrix(
    matrix: np.ndarray | list[list[int]],
    class_names: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Save a confusion matrix as a PNG image.

    Parameters
    ----------
    matrix:
        Confusion matrix values.

    class_names:
        Ordered diagnostic class names.

    output_path:
        Destination PNG path.

    title:
        Plot title.
    """

    matrix = np.asarray(
        matrix,
        dtype=int,
    )

    expected_shape = (
        len(class_names),
        len(class_names),
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected confusion matrix shape "
            f"{expected_shape}, but received {matrix.shape}."
        )

    output_path = Path(
        output_path
    )

    ensure_directory(
        output_path.parent
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(
            len(class_names)
        ),
        yticks=np.arange(
            len(class_names)
        ),
        xticklabels=class_names,
        yticklabels=class_names,
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size > 0
        else 0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index,
                    ]
                ),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if matrix[
                        row_index,
                        column_index,
                    ] > threshold
                    else "black"
                ),
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


def plot_training_history(
    history: dict[str, list[float]],
    output_directory: str | Path,
    fold_number: int,
) -> None:
    """
    Save training and validation curves.

    Expected history keys:

    train_loss
    validation_loss
    train_accuracy
    validation_accuracy
    train_f1
    validation_f1
    """

    output_directory = ensure_directory(
        output_directory
    )

    epochs = range(
        1,
        len(
            history.get(
                "train_loss",
                [],
            )
        ) + 1,
    )

    if not list(epochs):
        return

    # Loss curve.
    loss_figure, loss_axis = plt.subplots(
        figsize=(8, 5)
    )

    loss_axis.plot(
        epochs,
        history["train_loss"],
        label="Training loss",
    )

    loss_axis.plot(
        epochs,
        history["validation_loss"],
        label="Validation loss",
    )

    loss_axis.set_title(
        f"Fold {fold_number}: Loss"
    )

    loss_axis.set_xlabel(
        "Epoch"
    )

    loss_axis.set_ylabel(
        "Cross-entropy loss"
    )

    loss_axis.legend()

    loss_axis.grid(
        alpha=0.3
    )

    loss_figure.tight_layout()

    loss_figure.savefig(
        output_directory
        / f"fold_{fold_number}_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        loss_figure
    )

    # Accuracy curve.
    accuracy_figure, accuracy_axis = plt.subplots(
        figsize=(8, 5)
    )

    accuracy_axis.plot(
        epochs,
        history["train_accuracy"],
        label="Training accuracy",
    )

    accuracy_axis.plot(
        epochs,
        history["validation_accuracy"],
        label="Validation accuracy",
    )

    accuracy_axis.set_title(
        f"Fold {fold_number}: Accuracy"
    )

    accuracy_axis.set_xlabel(
        "Epoch"
    )

    accuracy_axis.set_ylabel(
        "Accuracy"
    )

    accuracy_axis.legend()

    accuracy_axis.grid(
        alpha=0.3
    )

    accuracy_figure.tight_layout()

    accuracy_figure.savefig(
        output_directory
        / f"fold_{fold_number}_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        accuracy_figure
    )

    # Macro F1-score curve.
    f1_figure, f1_axis = plt.subplots(
        figsize=(8, 5)
    )

    f1_axis.plot(
        epochs,
        history["train_f1"],
        label="Training macro F1",
    )

    f1_axis.plot(
        epochs,
        history["validation_f1"],
        label="Validation macro F1",
    )

    f1_axis.set_title(
        f"Fold {fold_number}: Macro F1-score"
    )

    f1_axis.set_xlabel(
        "Epoch"
    )

    f1_axis.set_ylabel(
        "Macro F1-score"
    )

    f1_axis.legend()

    f1_axis.grid(
        alpha=0.3
    )

    f1_figure.tight_layout()

    f1_figure.savefig(
        output_directory
        / f"fold_{fold_number}_f1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        f1_figure
    )
