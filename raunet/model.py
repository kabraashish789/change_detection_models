import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual  # Residual connection
        out = self.relu(out)
        return out

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class RAUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(RAUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.Conv1 = ResidualConvBlock(in_channels, 64)
        self.Conv2 = ResidualConvBlock(64, 128)
        self.Conv3 = ResidualConvBlock(128, 256)
        self.Conv4 = ResidualConvBlock(256, 512)
        self.Conv5 = ResidualConvBlock(512, 1024)

        # Decoder
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ResidualConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ResidualConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ResidualConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ResidualConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        e1 = self.Conv1(x)  # [B, 64, H, W]

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)  # [B, 128, H/2, W/2]

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)  # [B, 256, H/4, W/4]

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)  # [B, 512, H/8, W/8]

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)  # [B, 1024, H/16, W/16]

        # Decoding + Concatenation with Attention
        d5 = self.Up5(e5)  # [B, 512, H/8, W/8]
        s4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((s4, d5), dim=1)  # [B, 1024, H/8, W/8]
        d5 = self.UpConv5(d5)  # [B, 512, H/8, W/8]

        d4 = self.Up4(d5)  # [B, 256, H/4, W/4]
        s3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((s3, d4), dim=1)  # [B, 512, H/4, W/4]
        d4 = self.UpConv4(d4)  # [B, 256, H/4, W/4]

        d3 = self.Up3(d4)  # [B, 128, H/2, W/2]
        s2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((s2, d3), dim=1)  # [B, 256, H/2, W/2]
        d3 = self.UpConv3(d3)  # [B, 128, H/2, W/2]

        d2 = self.Up2(d3)  # [B, 64, H, W]
        s1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((s1, d2), dim=1)  # [B, 128, H, W]
        d2 = self.UpConv2(d2)  # [B, 64, H, W]

        out = self.Conv(d2)  # [B, out_channels, H, W]
        return out

def test():
    x = torch.randn(3, 6, 256, 256)  # Batch size 3, 6 input channels, 256x256 images
    model = RAUNet(in_channels=6, out_channels=1)
    preds = model(x)
    assert preds.shape == (3, 1, 256, 256)
    print("Model output shape:", preds.shape)

if __name__ == "__main__":
    test()
