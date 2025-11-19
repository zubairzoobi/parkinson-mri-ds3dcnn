import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

from model import NiftiDataset, Simple3DCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_path, data_path):
    dataset = NiftiDataset(data_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = Simple3DCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds, labels_list = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_list, preds)
    precision = precision_score(labels_list, preds, average="macro", zero_division=0)
    recall = recall_score(labels_list, preds, average="macro", zero_division=0)
    f1 = f1_score(labels_list, preds, average="macro", zero_division=0)
    roc_auc = roc_auc_score(np.eye(3)[labels_list], np.eye(3)[preds], average="macro")

    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC-AUC:", roc_auc)


if __name__ == "__main__":
    evaluate("PATH_TO_MODEL.pth", "PATH_TO_DATA")
