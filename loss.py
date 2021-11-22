# L1 Loss
# Perceptual Loss
# GAN Loss
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models

class ContentLoss(nn.Module):

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        # Extract the output of the thirty-fifth layer in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Standardized operations.
        sr = sr / 255
        hr = hr / 255
        # Find the feature map difference between the two images.
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss