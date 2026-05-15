"""
Run this after training to visually check if the model is working.
It shows the original photo, the true mask, and the model's prediction side by side.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import LightUNet

WEIGHTS_PATH = r".\test\w2\train2.pth"

frame = "frame_000059_20260303_080042_163470"
TEST_IMAGE   = fr"data\images\{frame}.png"   # <-- paste any image filename from your folder here
TEST_MASK    = fr"data\masks\{frame}.jpg"    # <-- paste the matching mask filename here

CLASS_COLORS = np.array([
    [0,   255, 0  ],  # 0 background  → green
    [100, 100, 100],  # 1 road        → gray
    [255, 255, 255],  # 2 white lane  → white
    [255, 255, 0  ],  # 3 yellow lane → yellow
    [0,   0,   255],  # 4 vehicle     → blue
], dtype=np.uint8)

# Load model
model = LightUNet(num_classes=5)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

# Load and preprocess image
img = Image.open(TEST_IMAGE).resize((320, 240))
img_np = np.array(img)
img_t  = torch.tensor((img_np / 255.0).transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

# Load true mask
true_mask = Image.open(TEST_MASK).resize((320, 240), Image.NEAREST)

# Run prediction
with torch.no_grad():
    pred_classes = model(img_t).argmax(dim=1).squeeze().numpy()  # (H, W)

# Convert class indices → colors for display
pred_colored = CLASS_COLORS[pred_classes]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img_np);         axes[0].set_title("Input photo");       axes[0].axis("off")
axes[1].imshow(true_mask);      axes[1].set_title("True mask (label)"); axes[1].axis("off")
axes[2].imshow(pred_colored);   axes[2].set_title("Model prediction");  axes[2].axis("off")
plt.tight_layout()
plt.savefig("prediction_check.png", dpi=150)
plt.show()
print("Saved to prediction_check.png")
