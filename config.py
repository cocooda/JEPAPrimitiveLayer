import torch

EMBED_DIM = 128
PATCH_SIZE = 4
IMAGE_H = 32
IMAGE_W = 32
TOKEN_DIM = 3 * PATCH_SIZE * PATCH_SIZE
ACTION_DIM = 4
MASK_RATIO = 0.15
VICREG_WEIGHT = 0.1
DRIFT_WEIGHT = 0.05
JEPA_WEIGHT = 1.0
EMA_DECAY = 0.99
BATCH_SIZE = 8
NUM_STEPS = 50
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "/kaggle/input/test1t/exported_maps"
