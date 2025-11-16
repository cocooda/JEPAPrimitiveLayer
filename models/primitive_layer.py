import torch
import torch.nn as nn
from utils.mask import random_token_mask
from utils.losses import jepa_masked_mse, vicreg_loss, drift_loss
from utils.ema_buffer import ema_update, init_target_from_online, LatentBuffer
from .patch_embed import PatchEmbed
from .token_encoder import TokenMLPEncoder
from .action_encoder import ActionEncoder
from .spatial_predictor import SpatialPredictorCNN
from utils.patch_utils import patchify, unpatchify  # new

PATCH_SIZE = 4
TOKEN_DIM = 3 * PATCH_SIZE * PATCH_SIZE
EMBED_DIM = 128
ACTION_DIM = 4
MASK_RATIO = 0.15
VICREG_WEIGHT = 0.1
DRIFT_WEIGHT = 0.05
JEPA_WEIGHT = 1.0
EMA_DECAY = 0.99

class PrimitiveLayer(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, in_ch=3, embed_dim=EMBED_DIM, action_dim=ACTION_DIM, ema_decay=EMA_DECAY):
        super().__init__()
        # Learnable patch embedding (optional, can replace token_encoder input)
        self.patch_embed = PatchEmbed(in_ch=in_ch, embed_dim=embed_dim, patch_size=patch_size)
        # Token-level MLP encoder for raw patches
        self.token_encoder = TokenMLPEncoder(token_dim=TOKEN_DIM, embed_dim=embed_dim)
        # Action encoder
        self.action_encoder = ActionEncoder(input_dim=action_dim, embed_dim=embed_dim)
        # Predictor over token grid
        self.spatial_predictor = SpatialPredictorCNN(embed_dim=embed_dim)
        # Target (EMA) token encoder
        self.target_token_encoder = TokenMLPEncoder(token_dim=TOKEN_DIM, embed_dim=embed_dim)
        init_target_from_online(self.token_encoder, self.target_token_encoder)

        self.ema_decay = ema_decay
        self.buffer = LatentBuffer(embed_dim, ema_decay)

    def forward(self, imgs, actions, mask=None):
        device = imgs.device
        B = imgs.shape[0]

        # --- Step 1: patchify raw pixel tokens for JEPA ---
        raw_tokens, ph, pw = patchify(imgs, patch_size=PATCH_SIZE)  # (B, N, token_dim)
        N = raw_tokens.shape[1]

        # --- Step 2: create mask if not provided ---
        if mask is None:
            mask = random_token_mask(B, N, MASK_RATIO, device=device)

        # --- Step 3: masked tokens for online encoder ---
        masked_tokens = raw_tokens.clone()
        masked_tokens[mask] = 0.0
        z_spatial = self.token_encoder(masked_tokens)   # (B, N, D)

        # --- Step 4: encode actions and add to token embeddings ---
        z_action = self.action_encoder(actions)         # (B, D)
        z_action_tokens = z_action.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        z_c = z_spatial + z_action_tokens

        # --- Step 5: predictor over token grid ---
        s_c = self.spatial_predictor(z_c, ph, pw)

        # --- Step 6: target embeddings from full raw tokens ---
        with torch.no_grad():
            z_t = self.target_token_encoder(raw_tokens)
        ema_update(self.token_encoder, self.target_token_encoder, self.ema_decay)

        # --- Step 7: pooled representations for VICReg and drift ---
        pooled_z_c = z_c.mean(dim=1)
        pooled_z_t = z_t.mean(dim=1)

        # --- Step 8: compute losses ---
        loss_jepa = jepa_masked_mse(s_c, z_t, mask) * JEPA_WEIGHT
        loss_vic = vicreg_loss(pooled_z_c, pooled_z_t) * VICREG_WEIGHT
        loss_drift_val = drift_loss(pooled_z_c, self.buffer.get_prev_z_c()) * DRIFT_WEIGHT
        self.buffer.store_prev_z_c(pooled_z_c)

        total_loss = loss_jepa + loss_vic + loss_drift_val

        return s_c, total_loss, {"JEPA": loss_jepa.detach(), "VICReg": loss_vic.detach(), "Drift": loss_drift_val.detach()}
