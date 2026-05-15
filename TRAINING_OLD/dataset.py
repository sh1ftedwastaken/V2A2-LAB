import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Reference colors for each class
CLASS_COLORS = np.array([
    [0,   255, 0  ],  # 0 background
    [100, 100, 100],  # 1 road
    [255, 255, 255],  # 2 white lane
    [255, 255, 0  ],  # 3 yellow lane
    [0,   0,   255],  # 4 vehicle (blue - yours)
    [255, 0,   0  ],  # 5 vehicle (red - partner's)
], dtype=np.float32)

def mask_to_labels(mask_img):
    """Find the closest class color for every pixel."""
    mask = np.array(mask_img.convert("RGB"), dtype=np.float32)
    H, W = mask.shape[:2]
    pixels = mask.reshape(-1, 3)  # (H*W, 3)
    
    # For each pixel, compute distance to all 5 class colors
    diffs = pixels[:, None, :] - CLASS_COLORS[None, :, :]  # (H*W, 5, 3)
    dists = np.sum(diffs ** 2, axis=-1)                    # (H*W, 5)
    labels = np.argmin(dists, axis=-1)                     # (H*W,)
    labels[labels == 5] = 4  # merge red vehicle → same class as blue vehicle
    
    return labels.reshape(H, W).astype(np.int64)

class RobotDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        # Collect all image filenames and sort them so they always pair correctly
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir) if f.endswith(".png")
        ])
        self.mask_dir = mask_dir
        print(f"Dataset loaded: {len(self.image_paths)} samples found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path  = self.image_paths[idx]

        # Find the matching mask by replacing .jpg with .png
        base_name = os.path.basename(img_path).replace(".png", ".jpg")
        mask_path = os.path.join(self.mask_dir, base_name)

        # --- Load image ---
        # Resize to 320x240 (width x height) as the model expects (3, 240, 320)
        img = Image.open(img_path).resize((320, 240))
        img = np.array(img) / 255.0                          # normalize to 0-1
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)  # (3, H, W)

        # --- Load mask ---
        mask = Image.open(mask_path).resize((320, 240), Image.NEAREST)  # NEAREST keeps colors exact
        mask = torch.tensor(mask_to_labels(mask), dtype=torch.long)     # (H, W) with values 0-4
        
        # Random horizontal flip
        if random.random() > 0.5:
            img  = torch.flip(img,  dims=[2])
            mask = torch.flip(mask, dims=[1])

        # Random brightness shift
        if random.random() > 0.5:
            img = torch.clamp(img * random.uniform(0.7, 1.3), 0, 1)
            
        # Random vertical flip (less common but helps)
        if random.random() > 0.7:
            img  = torch.flip(img,  dims=[1])
            mask = torch.flip(mask, dims=[0])

        # Random contrast
        if random.random() > 0.5:
            mean = img.mean()
            img  = torch.clamp((img - mean) * random.uniform(0.8, 1.2) + mean, 0, 1)

        # Random gaussian noise (makes model more robust)
        if random.random() > 0.5:
            noise = torch.randn_like(img) * 0.02
            img   = torch.clamp(img + noise, 0, 1)

        return img, mask
