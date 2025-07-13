import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Basic Convolution Block
#############################################
class BasicConv2d(nn.Module):
    """
    A helper module that performs a 2D convolution followed by BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

#############################################
# Token Projection
#############################################
class TokenProjection(nn.Module):
    """
    Projects input feature maps to a desired embedding dimension and flattens them as tokens for the transformer.
    """
    def __init__(self, in_channels, embed_dim):
        super(TokenProjection, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        B, C, H, W = x.size()
        # Flatten spatial dims: (B, C, H*W) and then transpose to (H*W, B, C)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, H, W

#############################################
# Transformer Encoder Block
#############################################
class TransformerEncoderLayer(nn.Module):
    """
    A single transformer encoder layer.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # x shape: (tokens, B, embed_dim)
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    """
    A stack of transformer encoder layers.
    """
    def __init__(self, embed_dim, num_heads, depth, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#############################################
# Siamese Backbone for Multi-scale Feature Extraction
#############################################
class SiameseBackbone(nn.Module):
    """
    A simple CNN backbone that extracts features at multiple scales from an input image.
    This backbone is shared between the before and after images.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(SiameseBackbone, self).__init__()
        # First scale (high resolution)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # Second scale (mid resolution)
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        # Third scale (low resolution)
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Returns features at three scales.
        f1: high resolution, f2: mid resolution, f3: low resolution.
        """
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return f1, f2, f3

#############################################
# ChangeFormerV6 Network
#############################################
class ChangeFormerV6(nn.Module):
    """
    ChangeFormerV6: A transformer-based siamese network for change detection.
    
    This network processes two images (before and after) with a shared CNN backbone,
    computes absolute differences at multiple scales, refines the deepest features via
    a transformer encoder, and decodes them into a change detection map.
    """
    def __init__(self, in_channels=3, base_channels=64, embed_dim=256, num_heads=8, depth=4, num_classes=2):
        super(ChangeFormerV6, self).__init__()
        # Shared backbone for feature extraction.
        self.backbone = SiameseBackbone(in_channels, base_channels)
        
        # Token projection and transformer for the deepest features.
        self.token_proj = TokenProjection(in_channels=base_channels * 4, embed_dim=embed_dim)
        self.transformer = TransformerEncoder(embed_dim, num_heads, depth)
        
        # Decoder that upsamples and fuses multi-scale difference features.
        self.decoder3 = nn.Sequential(
            nn.Conv2d(embed_dim, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
    def forward(self, img_before, img_after):
        """
        Forward pass of the network.
        
        Args:
            img_before (torch.Tensor): Before image tensor of shape (B, 3, H, W)
            img_after (torch.Tensor): After image tensor of shape (B, 3, H, W)
        
        Returns:
            torch.Tensor: Change detection map of shape (B, num_classes, H, W)
        """
        # Extract multi-scale features from both images.
        f1_b, f2_b, f3_b = self.backbone(img_before)
        f1_a, f2_a, f3_a = self.backbone(img_after)
        
        # Compute feature differences at each scale.
        f1_diff = torch.abs(f1_b - f1_a)  # High-resolution difference.
        f2_diff = torch.abs(f2_b - f2_a)  # Mid-resolution difference.
        f3_diff = torch.abs(f3_b - f3_a)  # Low-resolution difference.
        
        # Process the deepest features with token projection and transformer.
        tokens, H, W = self.token_proj(f3_diff)  # tokens: (B, num_tokens, embed_dim)
        # Transformer expects input shape (num_tokens, B, embed_dim)
        tokens = tokens.permute(1, 0, 2)
        tokens = self.transformer(tokens)
        # Reshape tokens back to spatial feature maps.
        tokens = tokens.permute(1, 2, 0).contiguous().view(f3_diff.size(0), -1, H, W)
        
        # Decoder: upsample transformer output and fuse with higher resolution differences.
        d3 = self.decoder3(tokens)
        d3_up = F.interpolate(d3, size=f2_diff.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3_up, f2_diff], dim=1))
        d2_up = F.interpolate(d2, size=f1_diff.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, f1_diff], dim=1))
        
        # Final prediction.
        out = self.final_conv(d1)
        return out

#############################################
# Example Usage and Test
#############################################
if __name__ == '__main__':
    # Instantiate the model.
    model = ChangeFormerV6(in_channels=3, base_channels=64, embed_dim=256, num_heads=8, depth=4, num_classes=1)
    
    # Create example before and after images (batch size 1, 3 channels, 256x256 resolution).
    img_before = torch.randn(1, 3, 256, 256)
    img_after = torch.randn(1, 3, 256, 256)
    
    # Forward pass.
    output = model(img_before, img_after)
    print("Output shape:", output.shape)
