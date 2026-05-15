import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from model import LightUNet
from dataset import RobotDataset ####################### TEST IS NEW ########################################

# ── Settings ───────────────────────────────────────────────────────────────────
IMAGE_DIR   = "data/images"
MASK_DIR    = "data/masks"
NUM_CLASSES = 5
BATCH_SIZE  = 4
EPOCHS      = 500
LR          = 3e-4
SAVE_PATH   = "lightunet_trained.pth"
# ──────────────────────────────────────────────────────────────────────────────

def dice_loss(preds, targets, num_classes=5, eps=1e-6):
    preds_soft = torch.softmax(preds, dim=1)
    total = 0.0
    for cls in range(num_classes):
        pred_c   = preds_soft[:, cls]
        target_c = (targets == cls).float()
        inter    = (pred_c * target_c).sum()
        total   += 1 - (2 * inter + eps) / (pred_c.sum() + target_c.sum() + eps)
    return total / num_classes

# Load dataset
dataset    = RobotDataset(IMAGE_DIR, MASK_DIR)
val_size   = max(1, int(len(dataset) * 0.2))
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {train_size} | Val samples: {val_size}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()}")

model = LightUNet(num_classes=NUM_CLASSES).to(device)

# Tweaked weights based on what we see in predictions:
# - Background raised slightly to fix wall misclassification
# - Vehicle lowered slightly to reduce blobbing
class_weights = torch.tensor([
    0.45,   # background
    0.50,   # road
    8.00,   # white lane
    5.00,   # yellow lane
    6.00,   # vehicle
]).to(device)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# weight_decay adds L2 regularization — penalizes overconfident predictions
# which directly helps reduce bobby vehicle shapes
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# CosineAnnealingLR smoothly reduces learning rate over all 300 epochs
# better than ReduceLROnPlateau for long training runs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# ── Training loop ──────────────────────────────────────────────────────────────
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss  = loss_fn(preds, masks) + dice_loss(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds     = model(images)
            val_loss += loss_fn(preds, masks).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch:3d}/{EPOCHS}  |  train loss: {train_loss:.4f}  |  val loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  --> Best model saved! (val loss: {val_loss:.4f})")

print(f"\nTraining complete. Best model saved to: {SAVE_PATH}")