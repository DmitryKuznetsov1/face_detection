import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
import os
from PIL import Image
from tqdm import tqdm
import csv


class AgeRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, images_paths_and_targets, transform=None):
        self.images_paths_and_targets = images_paths_and_targets
        self.image_folder = image_folder
        self.transform = transform

    def __getitem__(self, index):
        image_name, target = self.images_paths_and_targets[index]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images_paths_and_targets)


class AgeRegressor(nn.Module):
    def __init__(self):
        super(AgeRegressor, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Evaluation")
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def get_images_paths_and_targets(csv_file_path):
    images_paths_and_targets = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=";")

        images_col, targets_col = 0, 4
        for row in csv_reader:
            image_name = row[images_col]
            target = float(row[targets_col])
            images_paths_and_targets.append((image_name, target))
    return images_paths_and_targets


def main():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=45),  # Наклон до 30 градусов
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_folder_path = r"/content/drive/MyDrive/Colab Notebooks/UTKFace_Dataset"
    csv_file_path = os.path.join(dataset_folder_path, r"utkface.csv")

    images_paths_and_targers = get_images_paths_and_targets(csv_file_path)
    train_dataset_list, test_dataset_list = train_test_split(images_paths_and_targers, test_size=0.2, random_state=42)

    train_dataset = AgeRegressionDataset(dataset_folder_path, train_dataset_list, transform)
    test_dataset = AgeRegressionDataset(dataset_folder_path, test_dataset_list, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AgeRegressor().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        test_loss = evaluate(model, test_dataloader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print()
        torch.save(model.state_dict(), fr"/content/drive/MyDrive/Colab Notebooks/"
                                       "age_regression/age_regression_model-"
                                       "{epoch}-{test_loss}.pth")


if __name__ == "__main__":
    main()
