import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import IMAGE_H, IMAGE_W, ACTION_DIM

class DrivingSceneDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.map_dir = os.path.join(data_root, "maps")
        self.meta_dir = os.path.join(data_root, "metadata")
        self.files = sorted([f for f in os.listdir(self.map_dir) if f.endswith(".png")])
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_H, IMAGE_W)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.map_dir, self.files[idx])
        json_name = self.files[idx].replace(".png", ".json")
        json_path = os.path.join(self.meta_dir, json_name)

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        vel = acc = yaw_rate = steer = 0.0
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                meta = json.load(f)
                vel = meta.get("velocity", 0.0)
                acc = meta.get("acceleration", 0.0)
                yaw_rate = meta.get("yaw_rate", 0.0)
                steer = meta.get("steering_angle", 0.0)

        kin = torch.tensor([vel, acc, yaw_rate, steer], dtype=torch.float32)
        return img_tensor, kin
