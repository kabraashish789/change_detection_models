import torch
import torch.nn as nn
import math

# --------------------------
# Patch Embedding Module
# --------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, patch_size=4):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, 3, H, W] => [B, embed_dim, H/patch_size, W/patch_size]
        return self.proj(x)

# --------------------------
# Basic Transformer Block
# --------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [B, N, embed_dim]
        h = x
        x = self.norm1(x)
        # nn.MultiheadAttention expects (sequence_length, batch, embed_dim)
        x = x.transpose(0, 1)
        attn_out, _ = self.attn(x, x, x)
        x = attn_out.transpose(0, 1)
        x = x + h  # Residual connection
        
        h2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h2  # Second residual connection
        return x

# --------------------------
# Downsampling Module
# --------------------------
class Downsample(nn.Module):
    def __init__(self, embed_dim):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, H, W):
        # x: [B, N, embed_dim] where N = H * W
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)  # Reshape back to spatial map
        x = self.conv(x)  # Downsample spatially by 2
        new_H, new_W = H // 2, W // 2
        x = x.flatten(2).transpose(1, 2)  # Flatten back to tokens: [B, new_H*new_W, C]
        return x, new_H, new_W

# --------------------------
# Cross-Attention Block
# --------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        """
        x: query features [B, N, embed_dim]
        y: key and value features [B, N, embed_dim]
        Updates x using cross-attention from y.
        """
        q = self.norm(x)
        kv = self.norm(y)
        q = q.transpose(0, 1)  # shape: [N, B, embed_dim]
        kv = kv.transpose(0, 1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.transpose(0, 1)
        return x + self.dropout(attn_out)

# --------------------------
# BIT_CD-Inspired Model with Multi-Scale Decoder
# --------------------------
class BITChangeDetector(nn.Module):
    def __init__(self,
                 in_channels=6,       # Input: two images concatenated (3+3)
                 out_channels=1,
                 img_size=256,        # image size is set to 256x256
                 patch_size=4,
                 embed_dim=128,
                 depths=[2, 2, 2, 2],  # Transformer block counts per stage
                 num_heads=4,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super(BITChangeDetector, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Each branch processes 3 channels
        self.patch_embed = PatchEmbed(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        
        # Calculate number of patches for the initial stage. For img_size=256 & patch_size=4: 256/4 = 64 --> 64x64 patches.
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # --------------------
        # Transformer Backbone with Four Stages
        # --------------------
        # Stage 1 (resolution: 64x64)
        self.stage1 = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depths[0])])
        self.down1 = Downsample(embed_dim)  # 64 -> 32
        # Stage 2 (resolution: 32x32)
        self.stage2 = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depths[1])])
        self.down2 = Downsample(embed_dim)  # 32 -> 16
        # Stage 3 (resolution: 16x16)
        self.stage3 = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depths[2])])
        self.down3 = Downsample(embed_dim)  # 16 -> 8
        # Stage 4 (resolution: 8x8)
        self.stage4 = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depths[3])])
        
        # --------------------
        # Cross-Attention Blocks for Fusing the Two Branches at Each Stage
        # --------------------
        self.cross_attn1 = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn2 = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn3 = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn4 = CrossAttentionBlock(embed_dim, num_heads, dropout)
        
        # --------------------
        # Multi-Scale Decoder
        # Feature resolutions per stage:
        #   - fuse1: 64x64, fuse2: 32x32, fuse3: 16x16, fuse4: 8x8.
        # We progressively upsample and fuse features.
        # --------------------
        self.decoder_conv1 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
    
    def forward_branch(self, x):
        """
        Process a single 3-channel image through patch embedding and transformer stages.
        Returns a list of (features, H, W) for each stage.
        """
        # x: [B, 3, H, W]
        x = self.patch_embed(x)  # [B, embed_dim, 64, 64] for 256x256 input.
        B, C, H, W = x.shape   # H = W = 64
        x = x.flatten(2).transpose(1, 2)  # [B, 64*64, embed_dim]
        x = x + self.pos_embed  # Add learnable positional embeddings.
        feats = []
        
        # Stage 1: Resolution 64x64
        for blk in self.stage1:
            x = blk(x)
        feats.append((x, H, W))
        x, H, W = self.down1(x, H, W)  # New size: 32x32
        
        # Stage 2: Resolution 32x32
        for blk in self.stage2:
            x = blk(x)
        feats.append((x, H, W))
        x, H, W = self.down2(x, H, W)  # New size: 16x16
        
        # Stage 3: Resolution 16x16
        for blk in self.stage3:
            x = blk(x)
        feats.append((x, H, W))
        x, H, W = self.down3(x, H, W)  # New size: 8x8
        
        # Stage 4: Resolution 8x8
        for blk in self.stage4:
            x = blk(x)
        feats.append((x, H, W))
        return feats
    
    def forward(self, x):
        # x: [B, 6, H, W] where H=W=256. Split into two 3-channel images.
        img1 = x[:, :3, :, :]  # "Before" image
        img2 = x[:, 3:, :, :]  # "After" image
        
        feats1 = self.forward_branch(img1)  # List of features per stage for t1.
        feats2 = self.forward_branch(img2)  # List of features per stage for t2.
        
        fused_feats = []
        # Fuse features from each stage with cross-attention and absolute difference.
        
        # Stage 1 (64x64)
        f1, H1, W1 = feats1[0]
        f2, _, _ = feats2[0]
        f1 = self.cross_attn1(f1, f2)
        fuse1 = torch.abs(f1 - f2)  # [B, N, embed_dim]
        fuse1 = fuse1.transpose(1, 2).reshape(-1, self.embed_dim, H1, W1)
        fused_feats.append(fuse1)
        
        # Stage 2 (32x32)
        f1, H2, W2 = feats1[1]
        f2, _, _ = feats2[1]
        f1 = self.cross_attn2(f1, f2)
        fuse2 = torch.abs(f1 - f2)
        fuse2 = fuse2.transpose(1, 2).reshape(-1, self.embed_dim, H2, W2)
        fused_feats.append(fuse2)
        
        # Stage 3 (16x16)
        f1, H3, W3 = feats1[2]
        f2, _, _ = feats2[2]
        f1 = self.cross_attn3(f1, f2)
        fuse3 = torch.abs(f1 - f2)
        fuse3 = fuse3.transpose(1, 2).reshape(-1, self.embed_dim, H3, W3)
        fused_feats.append(fuse3)
        
        # Stage 4 (8x8)
        f1, H4, W4 = feats1[3]
        f2, _, _ = feats2[3]
        f1 = self.cross_attn4(f1, f2)
        fuse4 = torch.abs(f1 - f2)
        fuse4 = fuse4.transpose(1, 2).reshape(-1, self.embed_dim, H4, W4)
        fused_feats.append(fuse4)
        
        # --------------------
        # Decoder: fuse and upsample multi-scale features.
        # Resolutions: fuse4 (8x8), fuse3 (16x16), fuse2 (32x32), fuse1 (64x64).
        # We'll upsample progressively and then add two extra upsampling stages to reach 256x256.
        # --------------------
        x_dec = fused_feats[-1]  # fuse4: 8x8
        x_dec = self.upsample(x_dec)   # 8x8 -> 16x16
        x_dec = torch.cat([x_dec, fused_feats[2]], dim=1)  # Concatenate with fuse3 (16x16)
        x_dec = self.activation(self.decoder_conv1(x_dec))
        
        x_dec = self.upsample(x_dec)   # 16x16 -> 32x32
        x_dec = torch.cat([x_dec, fused_feats[1]], dim=1)  # Concatenate with fuse2 (32x32)
        x_dec = self.activation(self.decoder_conv2(x_dec))
        
        x_dec = self.upsample(x_dec)   # 32x32 -> 64x64
        x_dec = torch.cat([x_dec, fused_feats[0]], dim=1)  # Concatenate with fuse1 (64x64)
        x_dec = self.activation(self.decoder_conv3(x_dec))
        
        # Two extra upsampling steps to go from 64x64 -> 256x256:
        x_dec = self.upsample(x_dec)   # 64x64 -> 128x128
        x_dec = self.upsample(x_dec)   # 128x128 -> 256x256
        
        x_out = self.decoder_conv4(x_dec)
        return x_out

# --------------------------
# Testing the Model
# --------------------------
def test():
    # Create a dummy tensor: batch size 3, 6-channel (two images), 256x256.
    x = torch.randn(3, 6, 256, 256)
    model = BITChangeDetector(
        in_channels=6, 
        out_channels=1, 
        img_size=256, 
        patch_size=4,
        embed_dim=64,
        depths=[2, 2, 2, 2],
        num_heads=4
    )
    preds = model(x)
    # Expected output shape: [3, 1, 256, 256]
    assert preds.shape == (3, 1, 256, 256), f"Got shape {preds.shape}"
    print("Model output shape:", preds.shape)

if __name__ == "__main__":
    test()
