import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch import nn


class EfficientNetRegressor(nn.Module):
    def __init__(self):
        """
        EfficientNet Regressor constructor.

        Initializes the EfficientNet model with a custom fully connected layer for age regression.

        The EfficientNet backbone is pretrained on ImageNet.
        """
        super(EfficientNetRegressor, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b5")
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> int:
        """
        Forward pass through the EfficientNet Regressor.

        Applies the necessary image transformations and passes the input through the EfficientNet model
        to predict the age.

        Args:
            x (np.ndarray): Input image tensor.

        Returns:
            int: Predicted age.
        """
        output = self.model(x)
        age_prediction = int(output.item())
        return age_prediction

    @staticmethod
    def transform(face: np.ndarray) -> torch.Tensor:
        """
        Apply image transformations to the input face.

        Args:
            face (np.ndarray): Input face image.

        Returns:
            torch.Tensor: Transformed and preprocessed image tensor.
        """
        tfms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        img = tfms(face).unsqueeze(0)
        return img
