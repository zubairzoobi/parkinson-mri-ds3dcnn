# Repository Assets

This directory is intended for figures and visual material used in the main repository documentation.

## Recommended Assets

Verified images that may be added here include:

```text
assets/
├── model_architecture.png
├── preprocessing_pipeline.png
├── cross_validation_workflow.png
├── summed_confusion_matrix.png
├── training_curves.png
└── gradcam_example.png
```

## Model Architecture Figure

A model architecture diagram may illustrate:

```text
Input MRI
193 × 229 × 193 × 1

        ↓

Depthwise Separable Conv3D
1 → 32 channels

        ↓

MaxPool3D and Dropout3D

        ↓

Depthwise Separable Conv3D
32 → 64 channels

        ↓

MaxPool3D and Dropout3D

        ↓

Depthwise Separable Conv3D
64 → 128 channels

        ↓

MaxPool3D

        ↓

Flatten and Dropout

        ↓

Fully Connected Layer

        ↓

NC | PD | Prodromal
```

## Cross-Validation Figure

The cross-validation workflow may illustrate:

```text
Complete dataset

        ↓

Stratified five-fold split

        ↓

One fold reserved for testing

        ↓

Remaining folds divided into
training and validation subsets

        ↓

Best model selected using
validation macro F1-score

        ↓

Final evaluation on the
held-out test fold
```

## Grad-CAM Images

Only anonymized and publication-safe Grad-CAM examples should be included.

Before uploading a Grad-CAM image, verify that it does not contain:

- patient names;
- patient identifiers;
- hospital accession numbers;
- dates of birth;
- institutional file paths;
- confidential clinical metadata.

## Image Quality

Recommended image properties:

- PNG format;
- readable labels;
- high resolution;
- consistent font size;
- white or transparent background;
- no patient-identifying information;
- no unsupported performance claims.

## Important Notice

Do not add placeholder figures that could be mistaken for genuine experiment results.

All confusion matrices, training curves, Grad-CAM visualizations, and architecture diagrams should accurately represent the documented model and verified experiment.
