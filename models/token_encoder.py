import torch.nn as nn

class TokenMLPEncoder(nn.Module):
    """Per-token MLP encoder"""
    def __init__(self, token_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(token_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, tokens):
        return self.fc(tokens)  # (B, N, D)
