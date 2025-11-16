import torch

MASK_RATIO = 0.15

def random_token_mask(B, N, mask_ratio=MASK_RATIO, device="cpu"):
    mask = torch.rand(B, N, device=device) < mask_ratio
    return mask
