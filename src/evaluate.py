from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NiftiDirectoryDataset
from src.train import create_model, evaluate_loader
from src.utils import (
    ensure_directory,
    load_config,
    load_model_checkpoint,
    plot_confusion_matrix,
    save_json,
    select_device,
    set_seed,
)


def main(
    config_path: str,
    checkpoint_path: str,
    data_directory: str | None = None,
    output_directory: str | None = None,
) -> None:
    """
    Evaluate a trained DS-3DCNN checkpoint.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    checkpoint_path:
        Path to a trained .pth checkpoint.

    data_directory:
        Optional dataset directory.

        When this argument is not provided, the dataset
        directory from the YAML configuration is used.

    output_directory:
        Optional directory for evaluation results.
    """

    config = load_config(
        config_path
    )

    set_seed(
        int(config["seed"])
    )

    device = select_device()

    class_names = list(
        config["data"]["classes"]
    )

    num_classes = int(
        config["model"]["num_classes"]
    )

    input_shape = tuple(
        int(value)
        for value in config["data"]["input_shape"]
    )

    if len(class_names) != num_classes:
        raise ValueError(
            "The number of class names must match "
            "model.num_classes."
        )

    selected_data_directory = (
        data_directory
        if data_directory is not None
        else config["data"]["directory"]
    )

    print(
        f"Evaluation dataset: "
        f"{selected_data_directory}"
    )

    dataset = NiftiDirectoryDataset(
        directory=selected_data_directory,
        classes=class_names,
        expected_shape=input_shape,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(
            config["training"]["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            config["data"]["num_workers"]
        ),
        pin_memory=bool(
            config["data"]["pin_memory"]
        ),
    )

    model = create_model(
        config
    ).to(device)

    checkpoint_information = (
        load_model_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_metrics = evaluate_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
    )

    labels = np.asarray(
        evaluation_metrics["labels"],
        dtype=int,
    )

    predictions = np.asarray(
        evaluation_metrics["predictions"],
        dtype=int,
    )

    print(
        "\nEvaluation results"
    )

    print(
        "=" * 60
    )

    print(
        f"Loss:      "
        f"{evaluation_metrics['loss']:.4f}"
    )

    print(
        f"Accuracy:  "
        f"{evaluation_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{evaluation_metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{evaluation_metrics['recall']:.4f}"
    )

    print(
        f"Macro F1:  "
        f"{evaluation_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{evaluation_metrics['roc_auc']:.4f}"
    )

    print(
        "\nClassification report"
    )

    print(
        classification_report(
            labels,
            predictions,
            labels=list(
                range(num_classes)
            ),
            target_names=class_names,
            zero_division=0,
        )
    )

    report_dictionary: dict[
        str,
        Any,
    ] = classification_report(
        labels,
        predictions,
        labels=list(
            range(num_classes)
        ),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    selected_output_directory = (
        output_directory
        if output_directory is not None
        else config["output"]["directory"]
    )

    evaluation_directory = ensure_directory(
        Path(selected_output_directory)
        / "evaluation"
    )

    checkpoint_name = Path(
        checkpoint_path
    ).stem

    confusion_matrix_path = (
        evaluation_directory
        / (
            f"{checkpoint_name}_"
            f"confusion_matrix.png"
        )
    )

    plot_confusion_matrix(
        matrix=evaluation_metrics[
            "confusion_matrix"
        ],
        class_names=class_names,
        output_path=confusion_matrix_path,
        title=(
            "DS-3DCNN Evaluation "
            "Confusion Matrix"
        ),
    )

    checkpoint_summary = {
        key: value
        for key, value
        in checkpoint_information.items()
        if key
        not in {
            "model_state_dict",
            "optimizer_state_dict",
        }
    }

    result_dictionary = {
        "checkpoint": str(
            checkpoint_path
        ),
        "dataset_directory": str(
            selected_data_directory
        ),
        "number_of_samples": len(
            dataset
        ),
        "class_names": class_names,
        "checkpoint_information": (
            checkpoint_summary
        ),
        "metrics": {
            "loss": float(
                evaluation_metrics["loss"]
            ),
            "accuracy": float(
                evaluation_metrics["accuracy"]
            ),
            "precision": float(
                evaluation_metrics["precision"]
            ),
            "recall": float(
                evaluation_metrics["recall"]
            ),
            "f1": float(
                evaluation_metrics["f1"]
            ),
            "roc_auc": float(
                evaluation_metrics["roc_auc"]
            ),
            "confusion_matrix": (
                evaluation_metrics[
                    "confusion_matrix"
                ]
            ),
        },
        "classification_report": (
            report_dictionary
        ),
        "labels": labels,
        "predictions": predictions,
        "probabilities": (
            evaluation_metrics[
                "probabilities"
            ]
        ),
    }

    results_path = (
        evaluation_directory
        / (
            f"{checkpoint_name}_"
            f"evaluation.json"
        )
    )

    save_json(
        data=result_dictionary,
        output_path=results_path,
    )

    print(
        f"\nEvaluation JSON saved to: "
        f"{results_path.resolve()}"
    )

    print(
        f"Confusion matrix saved to: "
        f"{confusion_matrix_path.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Parkinson MRI "
            "DS-3DCNN checkpoint."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help=(
            "Path to the YAML "
            "configuration file."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Path to the trained "
            ".pth checkpoint."
        ),
    )

    parser.add_argument(
        "--data-directory",
        default=None,
        help=(
            "Optional evaluation dataset directory. "
            "This overrides the directory in the "
            "configuration file."
        ),
    )

    parser.add_argument(
        "--output-directory",
        default=None,
        help=(
            "Optional directory where evaluation "
            "results will be saved."
        ),
    )

    arguments = parser.parse_args()

    main(
        config_path=arguments.config,
        checkpoint_path=arguments.checkpoint,
        data_directory=arguments.data_directory,
        output_directory=arguments.output_directory,
    )
