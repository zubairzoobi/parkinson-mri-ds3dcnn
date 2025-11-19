import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from model import Simple3DCNN


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def forward(self, x):
        return self.model(x)

    def backward(self, output, target_class):
        self.model.zero_grad()
        output[:, target_class].backward(retain_graph=True)

    def generate_heatmap(self, x, target_class):
        output = self.forward(x)
        self.backward(output, target_class)

        activations = self.model.activations.detach()
        gradients = self.model.gradients.detach()

        weights = torch.mean(gradients, dim=[2,3,4], keepdim=True)
        heatmap = torch.sum(activations * weights, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap = heatmap / torch.max(heatmap)

        return heatmap.squeeze().cpu().numpy()


def save_heatmap_as_nifti(heatmap, filename):
    nifti_img = nib.Nifti1Image(heatmap, affine=np.eye(4))
    nib.save(nifti_img, filename)
