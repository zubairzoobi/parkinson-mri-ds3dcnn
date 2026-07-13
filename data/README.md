# Dataset Directory

The original clinical MRI dataset is not included in this repository because of privacy, institutional, and data-use restrictions.

## Expected Local Dataset Structure

Create the dataset directory locally using the following structure:

```text
dataset/
├── NC/
│   ├── subject_001.nii
│   ├── subject_002.nii.gz
│   └── ...
├── PD/
│   ├── subject_001.nii
│   ├── subject_002.nii.gz
│   └── ...
└── Prodromal/
    ├── subject_001.nii
    ├── subject_002.nii.gz
    └── ...
```

## Supported File Formats

The data loader supports:

```text
.nii
.nii.gz
```

## Class Mapping

The class order used during training is:

```text
NC = 0
PD = 1
Prodromal = 2
```

This order must remain unchanged when loading trained model checkpoints.

## Input Requirements

All MRI volumes must:

- contain a three-dimensional brain MRI volume;
- have the same spatial orientation;
- have the same voxel dimensions;
- have the same spatial shape;
- be spatially preprocessed before model training.

The default model configuration expects:

```text
193 × 229 × 193
```

## Preprocessing Assumptions

The following spatial preprocessing steps were completed before model training:

- image registration;
- skull stripping;
- brain masking;
- resampling;
- spatial normalization.

The public repository performs:

- NIfTI loading;
- conversion to 32-bit floating-point format;
- percentile intensity clipping;
- subject-wise z-score normalization.

## Privacy Notice

Do not upload the following files to GitHub:

- patient MRI volumes;
- patient identifiers;
- restricted clinical metadata;
- private dataset credentials;
- institutional server paths;
- files that violate the dataset licence or data-use agreement.

The repository's `.gitignore` file excludes `.nii` and `.nii.gz` files to help prevent accidental uploads.
