"""
Run this after convert_masks.py to verify the conversion worked correctly.
It checks one mask and prints the class distribution.
"""
import os
import numpy as np
from PIL import Image

MASK_DIR = "data/masks_single"

files = [f for f in os.listdir(MASK_DIR) if f.endswith(".png")]
if not files:
    print("ERROR: No .png files found in data/masks_single/")
    print("Did you run convert_masks.py first?")
    exit()

# Check first file
sample = files[0]
mask   = np.array(Image.open(os.path.join(MASK_DIR, sample)))

print(f"Checking: {sample}")
print(f"Shape:    {mask.shape}  (should be (240, 320))")
print(f"dtype:    {mask.dtype}  (should be uint8)")
print(f"Unique values: {np.unique(mask)}  (should only contain 0,1,2,3,4)")
print()

CLASS_NAMES = ["background", "road", "white lane", "yellow lane", "vehicle"]
total = mask.size
print("Class distribution:")
for cls in range(5):
    count = (mask == cls).sum()
    pct   = count / total * 100
    bar   = "█" * int(pct / 2)
    print(f"  {cls} {CLASS_NAMES[cls]:<15} {pct:5.3f}%  {bar}")

print()
print(f"Total masks converted: {len(files)}")
print("If values look reasonable, you're good to train!")
