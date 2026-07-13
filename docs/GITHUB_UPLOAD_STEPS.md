# GitHub Setup and Upload Guide

This document explains how to download, configure, test, and update the Parkinson MRI DS-3DCNN repository.

## Repository URL

```text
https://github.com/zubairzoobi/parkinson-mri-ds3dcnn
```

---

## 1. Clone the Repository

Open PowerShell, Command Prompt, or a terminal and run:

```bash
git clone https://github.com/zubairzoobi/parkinson-mri-ds3dcnn.git
```

Enter the repository directory:

```bash
cd parkinson-mri-ds3dcnn
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Run the Automated Tests

```bash
pytest -q
```

A successful result should look similar to:

```text
7 passed
```

---

## 5. Configure the Dataset

Open:

```text
configs/default.yaml
```

Temporarily replace:

```yaml
directory: /absolute/path/to/dataset
```

with the local dataset directory.

Example on Windows:

```yaml
directory: D:/research/parkinson_dataset
```

Example on Linux:

```yaml
directory: /home/user/research/parkinson_dataset
```

The expected dataset structure is:

```text
dataset/
├── NC/
├── PD/
└── Prodromal/
```

Do not commit a private institutional server path to GitHub.

Before pushing changes, restore the generic path:

```yaml
directory: /absolute/path/to/dataset
```

---

## 6. Run Training

```bash
python -m src.train --config configs/default.yaml
```

Training outputs are saved inside:

```text
outputs/
```

The outputs directory is excluded from Git tracking because it may contain large model checkpoints and experiment files.

---

## 7. Run Evaluation

Windows PowerShell:

```powershell
python -m src.evaluate --config configs/default.yaml --checkpoint outputs/best_fold_1.pth
```

Linux or macOS:

```bash
python -m src.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/best_fold_1.pth
```

---

## 8. Generate Grad-CAM

Windows PowerShell:

```powershell
python -m src.gradcam --config configs/default.yaml --checkpoint outputs/best_fold_1.pth --image D:/path/to/anonymized_scan.nii.gz --target-class PD --output-prefix outputs/gradcam_pd
```

Linux or macOS:

```bash
python -m src.gradcam \
  --config configs/default.yaml \
  --checkpoint outputs/best_fold_1.pth \
  --image /path/to/anonymized_scan.nii.gz \
  --target-class PD \
  --output-prefix outputs/gradcam_pd
```

Valid target classes are:

```text
NC
PD
Prodromal
```

---

## 9. Check Changed Files

Before committing changes, run:

```bash
git status
```

Review every changed or untracked file.

Confirm that the following are not being uploaded:

- MRI volumes;
- patient identifiers;
- private metadata;
- institutional paths;
- trained checkpoints;
- environment files;
- restricted dataset content.

---

## 10. Commit Changes

Add the files:

```bash
git add .
```

Create a commit:

```bash
git commit -m "Describe the completed project update"
```

Push to GitHub:

```bash
git push origin main
```

---

## 11. Update the Local Repository

Before starting new work, download the latest repository changes:

```bash
git pull origin main
```

---

## 12. Correct a Repository Remote

Check the configured remote:

```bash
git remote -v
```

The remote should point to:

```text
https://github.com/zubairzoobi/parkinson-mri-ds3dcnn.git
```

To correct it, run:

```bash
git remote set-url origin https://github.com/zubairzoobi/parkinson-mri-ds3dcnn.git
```

---

## 13. Recommended Git Workflow

Use this sequence when updating the project:

```bash
git pull origin main
git status
pytest -q
git add .
git commit -m "Add a clear description of the update"
git push origin main
```

---

## Data Privacy Checklist

Before every push, verify that the commit does not contain:

- `.nii` or `.nii.gz` MRI files;
- subject names or identification numbers;
- dates of birth;
- hospital or clinical identifiers;
- private dataset credentials;
- personal server paths;
- large `.pth`, `.pt`, or `.ckpt` files;
- restricted clinical spreadsheets.

The public repository should contain code, documentation, configuration templates, and verified non-identifiable figures only.
