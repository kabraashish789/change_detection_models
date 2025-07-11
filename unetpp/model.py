import torch
import torch.nn as nn
import torch.nn.functional as F

# Double convolution block (same as your original ConvBlock)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) for detecting changes between image pairs.
    The encoder path downsamples the input, while the nested decoder path 
    densely connects feature maps from different levels.
    
    Parameters:
      - in_channels: typically 6 (for paired before/after images, e.g. 3+3)
      - num_classes: number of output channels (e.g. 1 for binary segmentation)
      - deep_supervision: if True, outputs from multiple decoder depths are returned
      - filters: a list defining the number of channels at each level
    """
    def __init__(self, in_channels=6, num_classes=1, deep_supervision=True, filters=[64, 128, 256, 512, 1024]):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Encoder
        self.conv0_0 = ConvBlock(in_channels, filters[0])     # X0,0
        self.conv1_0 = ConvBlock(filters[0], filters[1])        # X1,0
        self.conv2_0 = ConvBlock(filters[1], filters[2])        # X2,0
        self.conv3_0 = ConvBlock(filters[2], filters[3])        # X3,0
        self.conv4_0 = ConvBlock(filters[3], filters[4])        # X4,0

        # Decoder (nested dense skip-connections)
        # First level of decoder
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])
        
        # Second level
        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2])
        
        # Third level
        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1])
        
        # Fourth level (final nested stage)
        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0])
        
        # Final output convolution(s)
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder pathway
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder pathway with nested skip connections (using upsampling and concatenation)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # Final output
        if self.deep_supervision:
            # Return a list of outputs from different nested depths
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


def test():
    # Create a random tensor with shape [B, channels, H, W]
    x = torch.randn(3, 6, 256, 256)  # Batch size 3; 6 channels (before and after image concatenated)
    model = UNetPlusPlus(in_channels=6, num_classes=1, deep_supervision=False)
    preds = model(x)
    # The expected output shape is [B, num_classes, H, W]
    assert preds.shape == (3, 1, 256, 256), f"Unexpected output shape: {preds.shape}"
    print("Model output shape:", preds.shape)


if __name__ == "__main__":
    test()
