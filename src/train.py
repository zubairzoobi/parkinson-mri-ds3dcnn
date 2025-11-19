import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model import NiftiDataset, Simple3DCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_class_distribution(dataset, indices):
    classes = ['NC', 'PD', 'Prodromal']
    counts = {cls: 0 for cls in classes}
    for idx in indices:
        _, label = dataset[idx]
        counts[classes[label]] += 1
    return counts


def train_model():
    dataset = NiftiDataset("YOUR_DATA_PATH")

    labels = [dataset[i][1] for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    fold_num = 0
    all_results = []

    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        fold_num += 1

        train_data = Subset(dataset, train_idx)
        test_data = Subset(dataset, test_idx)

        train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=12, shuffle=False)

        model = Simple3DCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        patience = 20
        no_improve = 0

        for epoch in range(100):
            model.train()
            train_preds, train_labels_list = [], []

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels_list.extend(labels.cpu().numpy())

            train_acc = accuracy_score(train_labels_list, train_preds)

            # Validation
            model.eval()
            test_preds, test_labels_list = [], []

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    test_preds.extend(predicted.cpu().numpy())
                    test_labels_list.extend(labels.cpu().numpy())

            test_acc = accuracy_score(test_labels_list, test_preds)

            if test_acc > best_acc:
                best_acc = test_acc
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f"fold_{fold_num}.pth"))
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

            print(f"FOLD {fold_num} | EPOCH {epoch+1} | TRAIN ACC {train_acc:.3f} | TEST ACC {test_acc:.3f}")

        precision = precision_score(test_labels_list, test_preds, average="macro", zero_division=0)
        recall = recall_score(test_labels_list, test_preds, average="macro", zero_division=0)
        f1 = f1_score(test_labels_list, test_preds, average="macro", zero_division=0)
        roc_auc = roc_auc_score(np.eye(3)[test_labels_list], np.eye(3)[test_preds], average="macro")

        all_results.append([best_acc, precision, recall, f1, roc_auc])

    print("\n=== FINAL CROSS-VALIDATION RESULTS ===")
    all_results = np.array(all_results)
    means = all_results.mean(axis=0)
    stds = all_results.std(axis=0)

    for name, mean, sd in zip(["ACC", "Precision", "Recall", "F1", "ROC-AUC"], means, stds):
        print(f"{name}: {mean:.4f} ± {sd:.4f}")


if __name__ == "__main__":
    train_model()
