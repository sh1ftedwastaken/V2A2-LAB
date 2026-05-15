"""
Run this to check what colors are actually in your mask files.
This tells us if the color mapping is correct before retraining.
"""
import os
import numpy as np
from PIL import Image
from collections import Counter

MASK_DIR = "data/masks"

all_colors = Counter()

mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(".jpg")][:30]  # check first 30

for filename in mask_files:
    path = os.path.join(MASK_DIR, filename)
    mask = np.array(Image.open(path).convert("RGB"))
    
    # Sample every 10th pixel to keep it fast
    pixels = mask[::10, ::10].reshape(-1, 3)
    for pixel in pixels:
        all_colors[tuple(pixel)] += 1

print("=== Colors found in your masks (most common first) ===")
print(f"{'Color (R,G,B)':<25} {'Pixel count':<15} {'Likely class'}")
print("-" * 60)

KNOWN = {
    (0,   255, 0  ): "background (green)",
    (100, 100, 100): "road (gray)",
    (255, 255, 255): "white lane",
    (255, 255, 0  ): "yellow lane",
    (0,   0,   255): "vehicle (blue - yours)",
    (255, 0,   0  ): "vehicle (red - partner)",
}

for color, count in all_colors.most_common(15):
    label = KNOWN.get(color, "??? UNKNOWN - needs mapping!")
    print(f"{str(color):<25} {count:<15} {label}")

print("\n=== Pixel distribution (how much of each class) ===")
total = sum(all_colors.values())
for color, count in all_colors.most_common(15):
    label = KNOWN.get(color, "UNKNOWN")
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    print(f"{label:<30} {pct:5.1f}% {bar}")