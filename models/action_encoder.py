import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, a):
        return self.fc(a)  # (B, embed_dim)
