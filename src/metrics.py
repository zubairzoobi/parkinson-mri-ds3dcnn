from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int = 3,
) -> dict[str, Any]:
    """
    Calculate classification metrics from labels and probabilities.

    Parameters
    ----------
    labels:
        Ground-truth class labels with shape:

        [number_of_samples]

    probabilities:
        Softmax probabilities with shape:

        [number_of_samples, number_of_classes]

    num_classes:
        Total number of diagnostic classes.

    Returns
    -------
    dict:
        Dictionary containing accuracy, macro precision,
        macro recall, macro F1-score, ROC-AUC,
        confusion matrix, and predicted class labels.
    """

    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be a one-dimensional array, "
            f"but received shape {labels.shape}."
        )

    if probabilities.ndim != 2:
        raise ValueError(
            "probabilities must be a two-dimensional array, "
            f"but received shape {probabilities.shape}."
        )

    if len(labels) != len(probabilities):
        raise ValueError(
            "The number of labels must match the number "
            "of probability rows."
        )

    if probabilities.shape[1] != num_classes:
        raise ValueError(
            f"Expected probabilities for {num_classes} classes, "
            f"but received {probabilities.shape[1]} classes."
        )

    if len(labels) == 0:
        raise ValueError(
            "Metrics cannot be calculated for an empty dataset."
        )

    if np.any(labels < 0) or np.any(
        labels >= num_classes
    ):
        raise ValueError(
            "The labels contain a class index outside "
            "the valid class range."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "The probability array contains NaN "
            "or infinite values."
        )

    # Select the class with the highest predicted probability.
    predictions = probabilities.argmax(
        axis=1
    )

    results: dict[str, Any] = {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(
                range(num_classes)
            ),
        ).tolist(),
        "predictions": predictions,
    }

    # Convert integer labels into one-hot encoded labels
    # for multiclass one-vs-rest ROC-AUC calculation.
    one_hot_labels = np.eye(
        num_classes,
        dtype=np.float32,
    )[labels]

    try:
        results["roc_auc"] = float(
            roc_auc_score(
                one_hot_labels,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )

    except ValueError:
        # ROC-AUC cannot be calculated when one or more
        # classes are missing from a small validation split.
        results["roc_auc"] = float(
            "nan"
        )

    return results
