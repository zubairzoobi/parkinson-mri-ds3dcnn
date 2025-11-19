# Classification-of-Parkinsons-Disease-using-DS3DCNNs
This project presents a deep learning–based method for detecting Parkinson’s disease from brain MRI using a custom depthwise separable 3D Convolutional Neural Network (DS-3DCNN). It was developed as part of my PhD work and evaluated on a multi-centre MRI dataset with three diagnostic groups: Normal Control (NC), Prodromal, and Parkinson’s Disease (PD).

**📌 Objective**
Develop a 3-class MRI-based model that identifies early Parkinson’s disease, with emphasis on distinguishing prodromal subjects from NC and PD by learning spatial patterns directly from 3D MRI volumes.

**🧠 Methodology**

**Model Architecture**
Depthwise separable 3D CNN designed for volumetric MRI.

Key elements:

• 35-layer architecture
• Two depthwise separable 3D convolutional blocks (32 and 64 filters)
• ReLU activations
• 3D max-pooling layers
• Fully connected classifier with three outputs (NC, Prodromal, PD)
• Softmax for final probability distribution

**⚙️ Training Workflow**
Step 1: Implement the DS-3DCNN architecture in PyTorch.
Step 2: Apply 5-fold cross-validation.
Step 3: For each fold train on four folds and validate on one.
Step 4: Monitor validation metrics and use early stopping.
Step 5: Use Grad-CAM to visualise class-relevant regions.
Step 6: Report mean performance and standard deviation across folds.

**🎯 Dataset**
Total scans: 426 T1-weighted brain MRIs.
Subjects divided into three groups:
• 142 NC
• 142 Prodromal
• 142 PD
All subjects from baseline visits within a multicentre longitudinal Parkinson’s cohort.
Each sample is treated as a 3D volume for model input.

**⚙️ Training Details**
Framework: PyTorch
Task: 3-class classification
Cross-validation: 5-fold
Optimizer: Adam
Learning rate: 1e-4
Loss: Cross-Entropy
Epochs: up to 100 with early stopping
Batch size: 16
Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

**📊 Results**
Mean performance across 5 folds:
• Accuracy ≈ 91% (±2%)
• Precision ≈ 0.92
• Recall ≈ 0.91
• F1-score ≈ 0.91
• ROC-AUC ≈ 0.93
The model generalises well and distinguishes NC, Prodromal, and PD consistently.

**Interpretability:**
Grad-CAM highlighted spatial regions used by the model for PD discrimination, supporting model trustworthiness and clinical interpretability.

**🛠️Technologies**
Language: Python
Frameworks: PyTorch, NumPy, scikit-learn
Tools: Anaconda, Spyder, Git, GPU/HPC environments
