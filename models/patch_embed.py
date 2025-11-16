import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings via Conv2d"""
    def __init__(self, in_ch=3, embed_dim=128, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, N, D), h, w
        """
        x = self.proj(x)             # (B, embed_dim, H/ps, W/ps)
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D) where N = h*w
        return x, h, w
