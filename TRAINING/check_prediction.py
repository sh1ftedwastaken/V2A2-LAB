"""
Test your trained model visually.
Shows: input photo | true mask | model prediction side by side.
Usage: python check_prediction.py
       python check_prediction.py --image data/images/frame_000018.png
       python check_prediction.py --num_images 10
"""
import argparse
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from torchvision import transforms
import torch.nn as nn

from train2 import build_model

NUM_CLASSES = 5
CLASS_NAMES = ["background", "road", "white lane", "yellow lane", "vehicle"]
CLASS_COLORS = np.array([
    [0,   255, 0  ],
    [100, 100, 100],
    [255, 255, 255],
    [255, 255, 0  ],
    [0,   0,   255],
], dtype=np.uint8)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                  nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
        if drop > 0: layers.append(nn.Dropout2d(drop))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class LightUNet(nn.Module):
    def __init__(self, num_classes=5, dropout_p=0.3):
        super().__init__()
        self.stem=DoubleConv(3,16); self.enc1=DoubleConv(16,32); self.pool1=nn.AvgPool2d(2,2)
        self.enc2=DoubleConv(32,64,dropout_p); self.pool2=nn.AvgPool2d(2,2)
        self.enc3=DoubleConv(64,128,dropout_p); self.pool3=nn.AvgPool2d(2,2)
        self.bottleneck=DoubleConv(128,128,dropout_p)
        self.up1=nn.ConvTranspose2d(128,64,2,stride=2); self.dec1=DoubleConv(192,64,dropout_p)
        self.up2=nn.ConvTranspose2d(64,32,2,stride=2);  self.dec2=DoubleConv(96,32)
        self.up3=nn.ConvTranspose2d(32,16,2,stride=2);  self.dec3=DoubleConv(48,16)
        self.head=nn.Conv2d(16,num_classes,1)
    def forward(self, x):
        x0=self.stem(x); s1=self.enc1(x0); x=self.pool1(s1)
        s2=self.enc2(x); x=self.pool2(s2); s3=self.enc3(x); x=self.pool3(s3)
        x=self.bottleneck(x)
        x=self.up1(x); x=torch.cat([x,s3],1); x=self.dec1(x)
        x=self.up2(x); x=torch.cat([x,s2],1); x=self.dec2(x)
        x=self.up3(x); x=torch.cat([x,s1],1); x=self.dec3(x)
        return self.head(x)

def load_model(weights_path, device, model_name="", encoder_weights=""):
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    resolved_model = model_name or ckpt_args.get("model", "lightunet")
    resolved_encoder_weights = encoder_weights or ckpt_args.get("encoder_weights", "imagenet")
    resolved_encoder_weights = None if resolved_encoder_weights == "none" else resolved_encoder_weights
    dropout_p = ckpt_args.get("dropout_p", 0.3)

    model = build_model(
        resolved_model,
        dropout_p=dropout_p,
        encoder_weights=resolved_encoder_weights,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, resolved_model

def predict(model, img_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img   = Image.open(img_path).convert("RGB").resize((320, 240))
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_t).argmax(dim=1).squeeze().cpu().numpy()
    return np.array(img), pred


def build_output_path(base_path, index, total):
    if total == 1:
        return base_path

    root, ext = os.path.splitext(base_path)
    ext = ext or ".png"
    return f"{root}_{index:02d}{ext}"


def load_sample(model, img_path, masks_dir, device):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(masks_dir, stem + ".png")
    has_mask = os.path.exists(mask_path)

    img_np, pred = predict(model, img_path, device)
    pred_colored = CLASS_COLORS[pred]

    sample = {
        "img_path": img_path,
        "img_np": img_np,
        "pred_colored": pred_colored,
        "has_mask": has_mask,
    }
    if has_mask:
        true_mask = np.array(Image.open(mask_path))
        sample["true_colored"] = CLASS_COLORS[np.clip(true_mask, 0, 4)]
    else:
        sample["true_colored"] = np.zeros_like(pred_colored)
    return sample


def draw_sample(fig, axes, sample, index, total):
    for ax in axes:
        ax.clear()

    axes[0].imshow(sample["img_np"])
    axes[0].set_title("Input photo")
    axes[0].axis("off")

    axes[1].imshow(sample["true_colored"])
    axes[1].set_title("True mask (label)" if sample["has_mask"] else "True mask (missing)")
    axes[1].axis("off")

    axes[2].imshow(sample["pred_colored"])
    axes[2].set_title("Model prediction")
    axes[2].axis("off")

    filename = os.path.basename(sample["img_path"])
    fig.suptitle(f"{index + 1}/{total}  {filename}", fontsize=12)
    fig.canvas.draw_idle()


def save_sample_figure(sample, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    draw_sample(fig, axes, sample, 0, 1)
    legend = [plt.Rectangle((0,0),1,1, color=np.array(CLASS_COLORS[i])/255)
              for i in range(NUM_CLASSES)]
    fig.legend(legend, CLASS_NAMES, loc="lower center", ncol=5, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",     default="")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--masks_dir", default="data/masks_single")
    p.add_argument("--weights",   default="checkpoints/best_resnet34_unet_ce_jaccard.pth")
    p.add_argument("--model",     default="",
                   help="Optional override. By default the script auto-detects the model from the checkpoint.")
    p.add_argument("--encoder_weights", default="",
                   help="Optional override for SMP backbones. Usually not needed when loading a checkpoint.")
    p.add_argument("--out",       default="predictions/prediction_check.png")
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find weights — check new location first, fallback to old
    if not os.path.exists(args.weights):
        fallback = "lightunet_trained.pth"
        if os.path.exists(fallback):
            args.weights = fallback
            print(f"Using old weights: {fallback}")
        else:
            print("ERROR: No model weights found! Train first: python train2.py")
            return

    model, model_name = load_model(args.weights, device, args.model, args.encoder_weights)
    print(f"Model loaded from: {args.weights}")
    print(f"Detected model: {model_name}")

    # Pick image(s)
    if args.image and os.path.exists(args.image):
        img_paths = [args.image]
    else:
        img_dir  = "data/images"
        all_imgs = sorted([f for f in os.listdir(img_dir)
                           if f.lower().endswith((".png",".jpg",".jpeg"))])
        if not all_imgs:
            print(f"ERROR: No images found in {img_dir}")
            return

        num_images = max(1, min(args.num_images, len(all_imgs)))
        if num_images < args.num_images:
            print(f"Requested {args.num_images} images, but only {len(all_imgs)} are available. Using {num_images}.")
        
        img_paths = [os.path.join(img_dir, name) for name in random.sample(all_imgs, num_images)] 
        print(f"Auto-selected {len(img_paths)} images.")

    samples = [load_sample(model, img_path, args.masks_dir, device) for img_path in img_paths]

    saved_paths = []
    os.makedirs(os.path.dirname("predictions/"), exist_ok=True)
    for idx, sample in enumerate(samples, start=1):
        out_path = build_output_path(args.out, idx, len(img_paths))
        save_sample_figure(sample, out_path)
        saved_paths.append(out_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    legend = [plt.Rectangle((0,0),1,1, color=np.array(CLASS_COLORS[i])/255)
              for i in range(NUM_CLASSES)]
    fig.legend(legend, CLASS_NAMES, loc="lower center", ncol=5, fontsize=9)
    plt.tight_layout()

    state = {"index": 0}

    def update(step):
        state["index"] = (state["index"] + step) % len(samples)
        draw_sample(fig, axes, samples[state["index"]], state["index"], len(samples))

    def on_key(event):
        if event.key in {"right", "down", "space"}:
            update(1)
        elif event.key in {"left", "up", "backspace"}:
            update(-1)

    ax_prev = fig.add_axes([0.35, 0.02, 0.12, 0.05])
    ax_next = fig.add_axes([0.53, 0.02, 0.12, 0.05])
    btn_prev = Button(ax_prev, "Previous")
    btn_next = Button(ax_next, "Next")
    btn_prev.on_clicked(lambda event: update(-1))
    btn_next.on_clicked(lambda event: update(1))

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_sample(fig, axes, samples[state["index"]], state["index"], len(samples))
    print("Viewer controls: left/right arrow keys, up/down, space, or the Previous/Next buttons.")
    plt.show()

    for path in saved_paths:
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
