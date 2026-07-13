from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from src.dataset import NiftiDirectoryDataset
from src.metrics import calculate_metrics
from src.model import DS3DCNN
from src.utils import (
    ensure_directory,
    load_config,
    load_model_checkpoint,
    plot_confusion_matrix,
    plot_training_history,
    save_checkpoint,
    save_json,
    select_device,
    set_seed,
)


def create_model(config: dict[str, Any]) -> DS3DCNN:
    """Create the DS-3DCNN model from the YAML configuration."""

    return DS3DCNN(
        num_classes=int(config["model"]["num_classes"]),
        input_shape=tuple(config["data"]["input_shape"]),
        dropout_3d=float(config["model"]["dropout_3d"]),
        dropout_fc=float(config["model"]["dropout_fc"]),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    gradient_clip_norm: float,
) -> dict[str, Any]:
    """
    Train the model for one epoch.

    Returns training loss and classification metrics.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    probabilities: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_samples += batch_size

        probabilities.append(
            F.softmax(
                logits.detach(),
                dim=1,
            ).cpu()
        )

        labels_list.append(
            labels.detach().cpu()
        )

    probability_array = torch.cat(
        probabilities
    ).numpy()

    label_array = torch.cat(
        labels_list
    ).numpy()

    metrics = calculate_metrics(
        labels=label_array,
        probabilities=probability_array,
        num_classes=num_classes,
    )

    metrics["loss"] = (
        total_loss
        / max(total_samples, 1)
    )

    return metrics


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """
    Evaluate the model on a validation or test DataLoader.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    probabilities: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_samples += batch_size

        probabilities.append(
            F.softmax(
                logits,
                dim=1,
            ).cpu()
        )

        labels_list.append(
            labels.cpu()
        )

    probability_array = torch.cat(
        probabilities
    ).numpy()

    label_array = torch.cat(
        labels_list
    ).numpy()

    metrics = calculate_metrics(
        labels=label_array,
        probabilities=probability_array,
        num_classes=num_classes,
    )

    metrics["loss"] = (
        total_loss
        / max(total_samples, 1)
    )

    metrics["labels"] = label_array
    metrics["probabilities"] = probability_array

    return metrics


def main(
    config_path: str,
) -> None:
    """
    Run stratified cross-validation training.
    """

    config = load_config(
        config_path
    )

    seed = int(
        config["seed"]
    )

    set_seed(seed)

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

    dataset = NiftiDirectoryDataset(
        directory=config["data"]["directory"],
        classes=class_names,
        expected_shape=input_shape,
    )

    all_labels = dataset.labels_array

    training = config[
        "training"
    ]

    output_directory = ensure_directory(
        config["output"]["directory"]
    )

    number_of_folds = int(
        training["number_of_folds"]
    )

    splitter = StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=seed,
    )

    fold_results: list[
        dict[str, Any]
    ] = []

    confusion_matrices: list[
        np.ndarray
    ] = []

    for fold, (
        train_validation_indices,
        test_indices,
    ) in enumerate(
        splitter.split(
            np.zeros(
                len(all_labels)
            ),
            all_labels,
        ),
        start=1,
    ):
        print(
            "\n" + "=" * 70
        )

        print(
            f"Fold {fold}/{number_of_folds}"
        )

        print(
            "=" * 70
        )

        train_indices, validation_indices = (
            train_test_split(
                train_validation_indices,
                test_size=float(
                    training[
                        "validation_fraction"
                    ]
                ),
                stratify=all_labels[
                    train_validation_indices
                ],
                random_state=seed + fold,
            )
        )

        loader_options = {
            "batch_size": int(
                training["batch_size"]
            ),
            "num_workers": int(
                config["data"][
                    "num_workers"
                ]
            ),
            "pin_memory": bool(
                config["data"][
                    "pin_memory"
                ]
            ),
        }

        generator = torch.Generator()

        generator.manual_seed(
            seed + fold
        )

        train_loader = DataLoader(
            Subset(
                dataset,
                train_indices,
            ),
            shuffle=True,
            generator=generator,
            **loader_options,
        )

        validation_loader = DataLoader(
            Subset(
                dataset,
                validation_indices,
            ),
            shuffle=False,
            **loader_options,
        )

        test_loader = DataLoader(
            Subset(
                dataset,
                test_indices,
            ),
            shuffle=False,
            **loader_options,
        )

        print(
            f"Training samples: "
            f"{len(train_indices)}"
        )

        print(
            f"Validation samples: "
            f"{len(validation_indices)}"
        )

        print(
            f"Test samples: "
            f"{len(test_indices)}"
        )

        model = create_model(
            config
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = Adam(
            model.parameters(),
            lr=float(
                training[
                    "learning_rate"
                ]
            ),
            weight_decay=float(
                training[
                    "weight_decay"
                ]
            ),
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(
                training[
                    "scheduler_factor"
                ]
            ),
            patience=int(
                training[
                    "scheduler_patience"
                ]
            ),
            min_lr=float(
                training[
                    "minimum_learning_rate"
                ]
            ),
        )

        checkpoint_path = (
            output_directory
            / f"best_fold_{fold}.pth"
        )

        best_validation_f1 = float(
            "-inf"
        )

        best_validation_loss = float(
            "inf"
        )

        best_epoch = 0
        patience_counter = 0

        history: dict[
            str,
            list[float],
        ] = {
            "train_loss": [],
            "validation_loss": [],
            "train_accuracy": [],
            "validation_accuracy": [],
            "train_f1": [],
            "validation_f1": [],
        }

        max_epochs = int(
            training["max_epochs"]
        )

        min_delta = float(
            training["min_delta"]
        )

        for epoch in range(
            1,
            max_epochs + 1,
        ):
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=num_classes,
                gradient_clip_norm=float(
                    training[
                        "gradient_clip_norm"
                    ]
                ),
            )

            validation_metrics = evaluate_loader(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
            )

            scheduler.step(
                validation_metrics[
                    "loss"
                ]
            )

            history[
                "train_loss"
            ].append(
                float(
                    train_metrics["loss"]
                )
            )

            history[
                "validation_loss"
            ].append(
                float(
                    validation_metrics[
                        "loss"
                    ]
                )
            )

            history[
                "train_accuracy"
            ].append(
                float(
                    train_metrics[
                        "accuracy"
                    ]
                )
            )

            history[
                "validation_accuracy"
            ].append(
                float(
                    validation_metrics[
                        "accuracy"
                    ]
                )
            )

            history[
                "train_f1"
            ].append(
                float(
                    train_metrics["f1"]
                )
            )

            history[
                "validation_f1"
            ].append(
                float(
                    validation_metrics[
                        "f1"
                    ]
                )
            )

            validation_f1 = float(
                validation_metrics[
                    "f1"
                ]
            )

            validation_loss = float(
                validation_metrics[
                    "loss"
                ]
            )

            improved = (
                validation_f1
                > best_validation_f1
                + min_delta
                or (
                    abs(
                        validation_f1
                        - best_validation_f1
                    )
                    <= min_delta
                    and validation_loss
                    < best_validation_loss
                )
            )

            if improved:
                best_validation_f1 = (
                    validation_f1
                )

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation_f1=validation_f1,
                    validation_loss=validation_loss,
                    output_path=checkpoint_path,
                )

            else:
                patience_counter += 1

            learning_rate = (
                optimizer
                .param_groups[0]["lr"]
            )

            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{max_epochs:03d} | "
                f"Train Loss "
                f"{train_metrics['loss']:.4f} | "
                f"Train Acc "
                f"{train_metrics['accuracy']:.4f} | "
                f"Train F1 "
                f"{train_metrics['f1']:.4f} | "
                f"Val Loss "
                f"{validation_metrics['loss']:.4f} | "
                f"Val Acc "
                f"{validation_metrics['accuracy']:.4f} | "
                f"Val F1 "
                f"{validation_metrics['f1']:.4f} | "
                f"Val AUC "
                f"{validation_metrics['roc_auc']:.4f} | "
                f"LR "
                f"{learning_rate:.8f}"
            )

            if patience_counter >= int(
                training["patience"]
            ):
                print(
                    f"Early stopping at "
                    f"epoch {epoch}. "
                    f"Best epoch: "
                    f"{best_epoch}."
                )

                break

        if best_epoch == 0:
            raise RuntimeError(
                f"No checkpoint was "
                f"saved for fold {fold}."
            )

        load_model_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        test_metrics = evaluate_loader(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        print(
            f"\nFold {fold} "
            f"test results"
        )

        print(
            classification_report(
                test_metrics["labels"],
                test_metrics[
                    "predictions"
                ],
                labels=list(
                    range(num_classes)
                ),
                target_names=class_names,
                zero_division=0,
            )
        )

        print(
            f"Accuracy="
            f"{test_metrics['accuracy']:.4f}, "
            f"Precision="
            f"{test_metrics['precision']:.4f}, "
            f"Recall="
            f"{test_metrics['recall']:.4f}, "
            f"F1="
            f"{test_metrics['f1']:.4f}, "
            f"ROC-AUC="
            f"{test_metrics['roc_auc']:.4f}"
        )

        confusion = np.asarray(
            test_metrics[
                "confusion_matrix"
            ],
            dtype=int,
        )

        confusion_matrices.append(
            confusion
        )

        plot_training_history(
            history=history,
            output_directory=output_directory,
            fold_number=fold,
        )

        plot_confusion_matrix(
            matrix=confusion,
            class_names=class_names,
            output_path=(
                output_directory
                / (
                    f"fold_{fold}_"
                    f"confusion_matrix.png"
                )
            ),
            title=(
                f"Confusion Matrix "
                f"- Fold {fold}"
            ),
        )

        fold_result = {
            "fold": fold,
            "best_epoch": best_epoch,
            "validation_f1": (
                best_validation_f1
            ),
            "validation_loss": (
                best_validation_loss
            ),
            "test_loss": float(
                test_metrics["loss"]
            ),
            "accuracy": float(
                test_metrics[
                    "accuracy"
                ]
            ),
            "precision": float(
                test_metrics[
                    "precision"
                ]
            ),
            "recall": float(
                test_metrics[
                    "recall"
                ]
            ),
            "f1": float(
                test_metrics["f1"]
            ),
            "roc_auc": float(
                test_metrics[
                    "roc_auc"
                ]
            ),
            "confusion_matrix": (
                confusion
            ),
        }

        fold_results.append(
            fold_result
        )

        save_json(
            data={
                "fold_result": (
                    fold_result
                ),
                "history": history,
            },
            output_path=(
                output_directory
                / (
                    f"fold_{fold}_"
                    f"results.json"
                )
            ),
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value
                in result.items()
                if key
                != "confusion_matrix"
            }
            for result
            in fold_results
        ]
    )

    results_frame.to_csv(
        output_directory
        / "cross_validation_results.csv",
        index=False,
    )

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    print(
        "\nCross-validation summary"
    )

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):
        values = results_frame[
            metric
        ].to_numpy(
            dtype=float
        )

        mean_value = float(
            np.nanmean(values)
        )

        standard_deviation = float(
            np.nanstd(values)
        )

        summary[metric] = {
            "mean": mean_value,
            "standard_deviation": (
                standard_deviation
            ),
        }

        print(
            f"{metric:<10}: "
            f"{mean_value:.4f} "
            f"± "
            f"{standard_deviation:.4f}"
        )

    summed_confusion = np.sum(
        confusion_matrices,
        axis=0,
    )

    plot_confusion_matrix(
        matrix=summed_confusion,
        class_names=class_names,
        output_path=(
            output_directory
            / (
                "summed_"
                "confusion_matrix.png"
            )
        ),
        title=(
            "Summed Confusion Matrix "
            "Across All Folds"
        ),
    )

    save_json(
        data={
            "summary": summary,
            "summed_confusion_matrix": (
                summed_confusion
            ),
        },
        output_path=(
            output_directory
            / (
                "cross_validation_"
                "summary.json"
            )
        ),
    )

    print(
        f"\nResults saved in: "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train the Parkinson MRI "
            "DS-3DCNN."
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

    arguments = parser.parse_args()

    main(
        arguments.config
    )
