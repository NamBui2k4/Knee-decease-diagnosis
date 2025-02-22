import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET, self).__init__()

        # Encoder
        self.conv1 = self.DoubleConv(in_channels, 64)
        self.conv2 = self.DoubleConv(64, 128)
        self.conv3 = self.DoubleConv(128, 256)
        self.conv4 = self.DoubleConv(256, 512)

        # Bottleneck
        self.conv5 = self.DoubleConv(512, 1024)

        # Decoder
        self.up6 = self.UpConv(1024, 512)
        self.conv6 = self.DoubleConv(1024, 512)

        self.up7 = self.UpConv(512, 256)
        self.conv7 = self.DoubleConv(512, 256)

        self.up8 = self.UpConv(256, 128)
        self.conv8 = self.DoubleConv(256, 128)

        self.up9 = self.UpConv(128, 64)
        self.conv9 = self.DoubleConv(128, 64)

        # Output Layer
        self.final = nn.Conv2d(64, out_channels, 1)

    def DoubleConv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def UpConv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def crop(self, tensor, target_tensor):
        """Cắt tensor để đảm bảo khớp kích thước với target_tensor"""
        _, _, h, w = target_tensor.shape
        return TF.center_crop(tensor, [h, w])

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = F.max_pool2d(c1, 2)

        c2 = self.conv2(p1)
        p2 = F.max_pool2d(c2, 2)

        c3 = self.conv3(p2)
        p3 = F.max_pool2d(c3, 2)

        c4 = self.conv4(p3)
        p4 = F.max_pool2d(c4, 2)

        c5 = self.conv5(p4)

        # Decoder
        u6 = self.up6(c5)
        c6 = self.conv6(torch.cat([u6, self.crop(c4, u6)], dim=1))

        u7 = self.up7(c6)
        c7 = self.conv7(torch.cat([u7, self.crop(c3, u7)], dim=1))

        u8 = self.up8(c7)
        c8 = self.conv8(torch.cat([u8, self.crop(c2, u8)], dim=1))

        u9 = self.up9(c8)
        c9 = self.conv9(torch.cat([u9, self.crop(c1, u9)], dim=1))

        output = self.final(c9)  # Không cần softmax (dùng trong loss function)

        return output

def test():
    x = torch.randn((1, 3, 161, 161))  # Batch size = 1, 3 channels, 161x161
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)  # In ra để kiểm tra kích thước đầu ra

if __name__ == "__main__":
    test()
