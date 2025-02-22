import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.datasets import ImageFolder


class CNN(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        # input: torch.Size([32, 3, 380, 380])
        self.conv1 = self.make_conv(3, 8,3) # torch.Size([32, 8, 126, 126])
        self.conv2 = self.make_conv(8, 16,3) # torch.Size([32, 16, 42, 42])
        self.conv3 = self.make_conv(16, 32,3) # torch.Size([32, 32, 14, 14])
        self.conv4 = self.make_conv(32, 64,3) # torch.Size([32, 64, 4, 4])

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.Dropout2d(p=0.5),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout2d(p=0.5),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout2d(p=0.5),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout2d(p=0.5),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Dropout2d(p=0.5),
            nn.ReLU()
        )
        self.fc6 = nn.Linear(16, num_class)

    def make_conv(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        b,c,h,w = x.size()
        x = x.view(b, c * h * w) # flatten

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

if __name__ == '__main__':
    model = CNN(3)