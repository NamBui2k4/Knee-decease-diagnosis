import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.datasets import ImageFolder

transform = Compose([
    ToTensor(),
    Resize((380, 380))
])

train_set = ImageFolder(root='dataset/train', transform=transform)
test_set = ImageFolder(root='dataset/test', transform=transform)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

if __name__ == '__main__':
    model = CNN()

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=32, # 281
        shuffle=True,
        drop_last=True
    )


    for images, labels in train_loader:
        predict = model(images)
        print(predict.shape)


