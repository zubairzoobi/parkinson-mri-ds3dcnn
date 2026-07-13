# Parkinson's Disease Classification from 3D Brain MRI using DS-3DCNN

A PyTorch-based deep learning project for classifying T1-weighted 3D brain MRI volumes into three diagnostic groups:

- Normal Control — NC
- Parkinson's Disease — PD
- Prodromal

This project was developed as part of my PhD research in Neuroscience and Imaging. It uses a custom Depthwise Separable 3D Convolutional Neural Network, referred to as DS-3DCNN, for volumetric MRI classification.

> This project is intended for research and educational purposes only. It is not a medical device and must not be used for clinical diagnosis.

---

## Project Objective

The objective of this project is to develop a three-class MRI-based classification model capable of distinguishing:

1. Normal Control subjects
2. Prodromal subjects
3. Parkinson's Disease patients

The prodromal group is particularly important because it represents an early or at-risk stage and is more difficult to distinguish from healthy controls and diagnosed Parkinson's Disease patients.

---

## Dataset

The research experiment used a balanced multicentre dataset containing 426 baseline T1-weighted brain MRI volumes.

| Diagnostic group | Number of MRI scans |
|---|---:|
| Normal Control | 142 |
| Parkinson's Disease | 142 |
| Prodromal | 142 |
| Total | 426 |

The original MRI dataset is not included in this repository because of privacy, institutional, and data-use restrictions.

The expected local dataset structure is:

```text
dataset/
├── NC/
│   ├── subject_001.nii
│   ├── subject_002.nii
│   └── ...
├── PD/
│   ├── subject_001.nii
│   ├── subject_002.nii
│   └── ...
└── Prodromal/
    ├── subject_001.nii
    ├── subject_002.nii
    └── ...
```

The class mapping used in the code is:

```text
NC = 0
PD = 1
Prodromal = 2
```

This class order must remain unchanged when loading trained checkpoints.

---

## MRI Preprocessing

Each MRI volume is processed using the following steps:

1. Load the NIfTI image using NiBabel.
2. Convert the volume to 32-bit floating-point format.
3. Clip intensity values between the 1st and 99th percentiles.
4. Add a single channel dimension.
5. Apply subject-wise z-score normalization.

The normalization is performed as:

```text
normalized_image = (image - mean) / (standard_deviation + 1e-8)
```

The current model expects MRI volumes with the spatial dimensions:

```text
193 × 229 × 193
```

The MRI registration, skull stripping, brain masking, resampling, and spatial normalization steps were completed before model training and are not included in this repository.

---

## Model Architecture

The proposed model is a Depthwise Separable 3D Convolutional Neural Network.

A depthwise separable 3D convolution consists of:

1. A depthwise 3D convolution that applies spatial filtering independently to every input channel.
2. A pointwise 1 × 1 × 1 convolution that combines information across channels.
3. Batch normalization.
4. ReLU activation.

The model contains three depthwise separable convolution blocks.

| Stage | Operation | Output channels |
|---|---|---:|
| Input | T1-weighted 3D MRI | 1 |
| Block 1 | Depthwise separable 3D convolution | 32 |
| Pooling | 3D max pooling and dropout | 32 |
| Block 2 | Depthwise separable 3D convolution | 64 |
| Pooling | 3D max pooling and dropout | 64 |
| Block 3 | Depthwise separable 3D convolution | 128 |
| Pooling | 3D max pooling | 128 |
| Classifier | Flatten, dropout, fully connected layer | 3 |

The spatial dimensions change after the three max-pooling operations as follows:

```text
193 × 229 × 193
→ 96 × 114 × 96
→ 48 × 57 × 48
→ 24 × 28 × 24
```

The final fully connected layer receives:

```text
128 × 24 × 28 × 24 features
```

and produces three output logits corresponding to NC, PD, and Prodromal classes.

---

## Training and Validation Strategy

The model is evaluated using stratified five-fold cross-validation.

For each fold:

1. One fold is reserved as the test set.
2. The remaining samples are divided into training and validation subsets.
3. The model is trained using the training subset.
4. The validation subset is used for early stopping and model selection.
5. The test set is evaluated only after the best validation-selected model has been restored.

The validation split is 15% of the training-development portion.

The best model is selected primarily using validation macro F1-score.

When validation F1-scores are approximately equal, the model with the lower validation loss is selected.

This design prevents the test fold from being used directly for model selection.

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Framework | PyTorch |
| Task | Three-class classification |
| Optimizer | Adam |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Loss function | Cross-entropy loss |
| Maximum epochs | 200 |
| Batch size | 8 |
| Cross-validation | Five-fold |
| Validation fraction | 15% |
| Early-stopping patience | 30 epochs |
| Gradient clipping | Maximum norm 1.0 |
| Scheduler | ReduceLROnPlateau |
| Random seed | 42 |

The learning-rate scheduler reduces the learning rate when the validation loss stops improving.

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Multiclass ROC-AUC
- Per-class precision
- Per-class recall
- Per-class F1-score
- Fold-wise confusion matrices
- Summed confusion matrix across all folds

ROC-AUC is calculated using softmax probabilities rather than hard predicted class labels.

---

## Reported Results

The following results were reported from the PhD experiment:

| Metric | Reported result |
|---|---:|
| Accuracy | Approximately 0.91 ± 0.02 |
| Macro precision | Approximately 0.92 |
| Macro recall | Approximately 0.91 |
| Macro F1-score | Approximately 0.91 |
| Macro ROC-AUC | Approximately 0.93 |

These results depend on the original MRI cohort, preprocessing pipeline, training environment, and cross-validation setup.

The original dataset is not publicly distributed, so exact reproduction requires authorized access to the same data and preprocessing procedure.

---

## Grad-CAM Interpretability

Grad-CAM is used to investigate which spatial regions contribute to a model prediction.

The Grad-CAM implementation:

- loads the same DS-3DCNN architecture used during training;
- loads a trained fold checkpoint;
- targets the third depthwise separable convolution block;
- calculates class-specific gradients;
- generates a three-dimensional activation map;
- resizes the map to the original MRI dimensions;
- saves the result as a NIfTI file;
- creates a two-dimensional center-slice overlay for visualization.

Grad-CAM outputs provide exploratory model interpretation and should not be interpreted as proof of biological or clinical causality.

---

## Repository Structure

```text
parkinson-mri-ds3dcnn/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── default.yaml
│
├── data/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── utils.py
│
├── tests/
│   └── test_model.py
│
├── assets/
│   └── README.md
│
└── results/
    └── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/zubairzoobi/parkinson-mri-ds3dcnn.git
```

Enter the project folder:

```bash
cd parkinson-mri-ds3dcnn
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Training

Before training, open:

```text
configs/default.yaml
```

and set the local dataset directory.

Then run:

```bash
python -m src.train --config configs/default.yaml
```

The training process saves:

- Best checkpoint for each fold
- Fold-wise confusion matrices
- Training and validation loss curves
- Accuracy and F1-score curves
- Cross-validation results CSV
- Summed confusion matrix

---

## Evaluation

A saved model checkpoint can be evaluated using:

```bash
python -m src.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/best_fold_1.pth
```

---

## Grad-CAM Generation

Generate a Grad-CAM visualization using:

```bash
python -m src.gradcam \
  --config configs/default.yaml \
  --checkpoint outputs/best_fold_1.pth \
  --image /path/to/anonymized_scan.nii.gz \
  --target-class PD \
  --output-prefix outputs/gradcam_pd
```

The command produces:

```text
outputs/gradcam_pd.nii.gz
outputs/gradcam_pd.png
```

Valid target classes are:

```text
NC
PD
Prodromal
```

---

## Limitations

- The original clinical MRI dataset is not publicly available.
- The model has not yet been validated on a completely independent external cohort.
- Results may be influenced by scanner, imaging site, demographic, and preprocessing differences.
- The current fully connected layer requires a fixed MRI input shape.
- Structural MRI alone may not capture every prodromal Parkinson's Disease characteristic.
- Grad-CAM provides approximate localization and does not establish biological causality.
- Prospective clinical validation would be required before real-world clinical use.

---

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- scikit-learn
- NiBabel
- Matplotlib
- Git and GitHub
- GPU and HPC environments

---

## Author

**Muhammad Zubair, PhD**

Medical AI and neuroimaging researcher

- GitHub: [zubairzoobi](https://github.com/zubairzoobi)
- LinkedIn: [Muhammad Zubair](https://www.linkedin.com/in/muhammad-zubair93/)
- Email: zubairzoobi93@gmail.com
