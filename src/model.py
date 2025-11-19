import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib

class NiftiDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.classes = ['NC', 'PD', 'Prodromal']
        self.files = []
        self.labels = []

        for i, cls in enumerate(self.classes):
            cls_folder = os.path.join(directory, cls)
            for fname in os.listdir(cls_folder):
                if fname.endswith('.nii'):
                    self.files.append(os.path.join(cls_folder, fname))
                    self.labels.append(i)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        nifti_image = nib.load(file_path)
        image = nifti_image.get_fdata()
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        image = (image - image.mean()) / image.std()
        return image, self.labels[idx]


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.point_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.batch_norm(x)
        return F.relu(x)


class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        self.layer1 = DepthwiseSeparableConv3D(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)

        self.layer2 = DepthwiseSeparableConv3D(32, 64, kernel_size=3, padding=1)
        self.layer3 = DepthwiseSeparableConv3D(64, 128, kernel_size=3, padding=1)

        self.fc = nn.Linear(128 * 24 * 28 * 24, 3)

        # Grad-CAM hooks
        self.gradients = None
        self.activations = None

        def hook_activations(module, input, output):
            self.activations = output

        def hook_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.layer3.register_forward_hook(hook_activations)
        self.layer3.register_full_backward_hook(hook_gradients)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)

        x = self.layer2(x)
        x = self.pool(x)

        x = self.layer3(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
