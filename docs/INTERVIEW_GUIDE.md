# Interview Guide

## 30-Second Project Introduction

This project was developed as part of my PhD research. Its objective is to classify T1-weighted three-dimensional brain MRI volumes into three groups: Normal Control, Parkinson's Disease, and Prodromal.

I implemented a custom depthwise separable 3D convolutional neural network in PyTorch. The complete pipeline includes MRI preprocessing, stratified five-fold cross-validation, validation-based model selection, test evaluation, checkpoint management, and three-dimensional Grad-CAM interpretation.

---

## Problem Statement

Parkinson's Disease is a progressive neurological disorder.

The purpose of this research project was to investigate whether structural information contained in T1-weighted brain MRI could help distinguish:

- Normal Control subjects;
- Parkinson's Disease patients;
- Prodromal subjects.

The prodromal group makes this a more difficult three-class classification problem because these subjects may show early characteristics without having a confirmed Parkinson's Disease diagnosis.

---

## Dataset

The research experiment used 426 baseline T1-weighted MRI volumes:

| Class | Number of subjects |
|---|---:|
| Normal Control | 142 |
| Parkinson's Disease | 142 |
| Prodromal | 142 |
| Total | 426 |

The dataset is not included in the public repository because it contains restricted clinical imaging data.

The class mapping is:

```text
NC = 0
PD = 1
Prodromal = 2
```

---

## Preprocessing

The spatial MRI preprocessing was completed before model training.

The public data loader performs the following operations:

1. Loads the NIfTI image.
2. Converts the image to float32.
3. Clips intensities between the 1st and 99th percentiles.
4. Applies subject-wise z-score normalization.
5. Adds the MRI channel dimension.
6. Verifies the expected spatial shape.

The expected input shape is:

```text
193 × 229 × 193
```

---

## Model Architecture

The model is a depthwise separable 3D convolutional neural network.

It contains three convolution blocks:

```text
Block 1: 1 → 32 channels
Block 2: 32 → 64 channels
Block 3: 64 → 128 channels
```

Every depthwise separable block contains:

1. Depthwise 3D convolution
2. Pointwise 1 × 1 × 1 convolution
3. Batch normalization
4. ReLU activation

The model then uses three max-pooling operations, dropout, flattening, and a fully connected layer for three-class classification.

---

## Why Depthwise Separable Convolution?

A standard 3D convolution can have a large computational and parameter cost.

Depthwise separable convolution divides the operation into two parts:

- spatial filtering independently for every input channel;
- channel combination using a pointwise convolution.

This reduces unnecessary computation while retaining three-dimensional spatial feature extraction.

---

## Training Strategy

The model uses stratified five-fold cross-validation.

For every fold:

1. One fold is reserved as the test set.
2. The remaining data is divided into training and validation subsets.
3. The model is trained only on the training subset.
4. The validation subset is used for early stopping and model selection.
5. The test fold is evaluated after loading the best validation-selected checkpoint.

The validation split is 15% of the training-development data.

The best checkpoint is selected primarily using validation macro F1-score. Validation loss is used as a tie-breaker.

---

## Main Training Settings

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.0003 |
| Weight decay | 0.00001 |
| Maximum epochs | 200 |
| Batch size | 8 |
| Cross-validation folds | 5 |
| Early-stopping patience | 30 |
| Gradient clipping | 1.0 |
| Random seed | 42 |
| Loss function | Cross-entropy |

---

## Evaluation Metrics

The pipeline calculates:

- accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- multiclass one-vs-rest ROC-AUC;
- classification report;
- per-fold confusion matrix;
- summed confusion matrix.

Macro averaging gives equal importance to every diagnostic class.

ROC-AUC is calculated using softmax probabilities rather than hard predicted labels.

---

## Grad-CAM

The project includes a checkpoint-compatible three-dimensional Grad-CAM implementation.

Grad-CAM:

1. Loads the same model architecture used for training.
2. Captures activations from the third convolution block.
3. Calculates gradients for a selected target class.
4. Combines the activations using gradient-based weights.
5. Resizes the heatmap to the original MRI dimensions.
6. Saves the complete heatmap as a NIfTI file.
7. Saves a center-slice overlay as a PNG image.

Grad-CAM is used for exploratory interpretation. It does not prove clinical or biological causality.

---

## Reported Research Results

The PhD experiment reported approximately:

| Metric | Result |
|---|---:|
| Accuracy | 0.91 ± 0.02 |
| Macro precision | 0.92 |
| Macro recall | 0.91 |
| Macro F1-score | 0.91 |
| Macro ROC-AUC | 0.93 |

These results depend on the original cohort, preprocessing procedure, data split, software environment, and hardware.

---

## Project Limitations

Important limitations include:

- no completely independent external validation cohort;
- possible scanner and site variability;
- restricted clinical dataset;
- fixed MRI input dimensions;
- structural MRI may not capture every prodromal characteristic;
- Grad-CAM provides approximate interpretation;
- the model is intended for research rather than clinical diagnosis.

---

## Common Interview Questions

### Why did you use macro F1-score?

This is a three-class problem, and each diagnostic class is clinically important. Macro F1-score calculates the F1-score separately for every class and then gives all classes equal importance.

### How did you avoid test-data leakage?

The test fold was not used for early stopping or model selection. Model selection was performed using a validation subset created only from the training-development portion.

### Why did you use five-fold cross-validation?

The dataset is limited in size. Five-fold cross-validation provides a more reliable performance estimate by testing the model on different held-out subsets.

### Why did you use softmax probabilities for ROC-AUC?

ROC-AUC requires continuous class scores or probabilities. Hard predicted labels discard confidence information and are not suitable for proper ROC-AUC calculation.

### Why is the dataset not included?

The MRI data is subject to privacy, institutional, and data-use restrictions. The public repository therefore contains only code and documentation.

### Is the model ready for clinical use?

No. It is a research prototype. External validation, prospective evaluation, regulatory review, and clinical testing would be required before clinical use.

### What would you improve next?

Possible improvements include:

- independent external validation;
- scanner and site harmonization;
- subject-grouped splitting for longitudinal data;
- multimodal imaging or clinical information;
- comparison with classical machine-learning baselines;
- uncertainty estimation;
- calibration analysis;
- additional explainability methods;
- deployment through a validated inference API.

---

## Two-Minute Project Explanation

This project addresses three-class classification of Normal Control, Prodromal, and Parkinson's Disease subjects from T1-weighted 3D brain MRI.

I developed the complete pipeline in PyTorch. The data loader reads NIfTI volumes, applies percentile-based intensity clipping and subject-wise z-score normalization, and verifies the expected MRI dimensions.

The model contains three depthwise separable 3D convolution blocks with 32, 64, and 128 output channels. Depthwise separable convolutions were used to reduce unnecessary computational cost compared with standard 3D convolutions.

For evaluation, I used stratified five-fold cross-validation. Within every fold, the training-development portion was divided into training and validation subsets. The validation macro F1-score was used for checkpoint selection and early stopping. The held-out test fold was evaluated only after restoring the best checkpoint.

The pipeline reports accuracy, macro precision, macro recall, macro F1-score, multiclass ROC-AUC, and confusion matrices. I also implemented three-dimensional Grad-CAM using the final convolution block to provide class-specific spatial interpretation.

The project is intended as a reproducible research implementation, not as a clinical diagnostic system. The public repository excludes all restricted clinical MRI data.
