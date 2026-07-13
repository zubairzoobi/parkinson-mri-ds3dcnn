# Original Research Code

This directory preserves the original research scripts used during the Parkinson MRI DS-3DCNN experiment.

The scripts are included for:

- research traceability;
- comparison with the cleaned implementation;
- preservation of the original experimental workflow;
- checkpoint architecture reference.

## Files

```text
original_code/
├── PD_latest_code_validation.py
└── grad_cam_original.py
```

## Important Notice

The scripts in this directory represent the original experimental implementation.

They may contain:

- hard-coded dataset paths;
- environment-specific settings;
- older coding structure;
- duplicated functions;
- dependencies on the original research environment.

For the cleaned and reusable implementation, use the files inside:

```text
src/
```

## Main Cleaned Commands

Training:

```bash
python -m src.train --config configs/default.yaml
```

Evaluation:

```bash
python -m src.evaluate \
  --config configs/default.yaml \
  --checkpoint outputs/best_fold_1.pth
```

Grad-CAM:

```bash
python -m src.gradcam \
  --config configs/default.yaml \
  --checkpoint outputs/best_fold_1.pth \
  --image /path/to/anonymized_scan.nii.gz \
  --target-class PD \
  --output-prefix outputs/gradcam_pd
```

## Privacy Warning

Before publishing original research scripts, verify that they do not expose:

- patient identifiers;
- private MRI paths;
- institutional server addresses;
- usernames or passwords;
- dataset credentials;
- confidential clinical metadata.

The original scripts should not be treated as the recommended production implementation.
