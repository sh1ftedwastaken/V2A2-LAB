"""
Run this ONCE to convert your color .jpg masks → single-channel .png masks
that work with the new training script.

Before running:
    - data/masks/   has your color .jpg files  (input)

After running:
    - data/masks_single/   has new .png files  (output)
    - each pixel value = class number (0-4) instead of a color

Then point train.py --masks_dir data/masks_single
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path

INPUT_DIR  = "data/masks"        # your current color .jpg masks
OUTPUT_DIR = "data/masks_single" # new single-channel .png masks

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference class colors — closest-match approach (handles slight color variations)
CLASS_COLORS = np.array([
    [0,   255, 0  ],  # 0 background   → green
    [100, 100, 100],  # 1 road         → gray (your actual value)
    [255, 255, 255],  # 2 white lane   → white
    [255, 255, 0  ],  # 3 yellow lane  → yellow
    [0,   0,   255],  # 4 vehicle blue → yours
    [255, 0,   0  ],  # 5 vehicle red  → partner's (will be remapped to 4)
], dtype=np.float32)

def color_mask_to_single_channel(mask_img):
    """Convert color mask → 2D array of class indices (0-4)."""
    mask = np.array(mask_img.convert("RGB"), dtype=np.float32)
    H, W = mask.shape[:2]
    pixels = mask.reshape(-1, 3)

    # Find closest class color for each pixel
    diffs  = pixels[:, None, :] - CLASS_COLORS[None, :, :]
    dists  = np.sum(diffs ** 2, axis=-1)
    labels = np.argmin(dists, axis=-1)

    # Merge red vehicle (index 5) → same class as blue vehicle (index 4)
    labels[labels == 5] = 4

    return labels.reshape(H, W).astype(np.uint8)

mask_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]
print(f"Found {len(mask_files)} mask files to convert...")

for i, filename in enumerate(mask_files):
    src_path = os.path.join(INPUT_DIR, filename)
    # Output filename: same stem but .png extension
    stem     = Path(filename).stem
    dst_path = os.path.join(OUTPUT_DIR, stem + ".png")

    mask_img   = Image.open(src_path)
    label_mask = color_mask_to_single_channel(mask_img)

    # Save as single-channel PNG (mode "L" = grayscale, values 0-255)
    Image.fromarray(label_mask, mode="L").save(dst_path)

    if (i + 1) % 50 == 0:
        print(f"  Converted {i+1}/{len(mask_files)}...")

print(f"\nDone! Converted {len(mask_files)} masks → data/masks_single/")
print("Verify one file looks right by running: python verify_conversion.py")
