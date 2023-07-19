from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from torch import nn


class AgeRegressor(nn.Module):
    def __init__(self):
        super(AgeRegressor, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.transform(x)
        output = self.model(x)
        age_prediction = int(output.item())
        return age_prediction

    @staticmethod
    def transform(face):
        tfms = transforms.Compose([transforms.ToTensor(), transforms.Resize(224),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        img = tfms(face).unsqueeze(0)
        return img