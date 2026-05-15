"""
Run trained segmentation model on raw robot images.

Outputs for each input image:
- a single-channel predicted mask (.png)
- a colorized label image (.png)
- an overlay of labels on top of the raw image (.png)

Examples:
    py robot_inference.py --input data_test/images/frame.png
    py robot_inference.py --input data_test/images --weights checkpoints/best_resnet34_unet_ce_jaccard.pth
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train2 import build_model

NUM_CLASSES = 5
CLASS_NAMES = ["background", "road", "white lane", "yellow lane", "vehicle"]
CLASS_COLORS = np.array([
    [0, 255, 0],
    [100, 100, 100],
    [255, 255, 255],
    [255, 255, 0],
    [0, 0, 255],
], dtype=np.uint8)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MODEL_INPUT_SIZE = (320, 240)


def parse_args():
    parser = argparse.ArgumentParser(description="Run robot image segmentation inference.")
    parser.add_argument("--input", required=True,
                        help="Path to a raw image or a folder of raw images.")
    parser.add_argument("--weights", default="checkpoints/best_resnet34_unet_ce_jaccard.pth",
                        help="Path to a trained checkpoint.")
    parser.add_argument("--output_dir", default="robot_outputs",
                        help="Folder where masks and labeled outputs will be saved.")
    parser.add_argument("--model", default="",
                        help="Optional model override. Usually auto-detected from the checkpoint.")
    parser.add_argument("--encoder_weights", default="",
                        help="Optional encoder override. Usually auto-detected from the checkpoint.")
    parser.add_argument("--overlay_alpha", type=float, default=0.45,
                        help="How strong the label overlay should be, from 0 to 1.")
    return parser.parse_args()


def load_model(weights_path: Path, device: torch.device, model_name: str, encoder_weights: str):
    checkpoint = torch.load(weights_path, map_location=device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    resolved_model = model_name or checkpoint_args.get("model", "lightunet")
    resolved_encoder_weights = encoder_weights or checkpoint_args.get("encoder_weights", "imagenet")
    resolved_encoder_weights = None if resolved_encoder_weights == "none" else resolved_encoder_weights
    dropout_p = checkpoint_args.get("dropout_p", 0.3)

    model = build_model(
        resolved_model,
        dropout_p=dropout_p,
        encoder_weights=resolved_encoder_weights,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, resolved_model


def list_input_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    image_paths = sorted(
        path for path in input_path.iterdir()
        if path.suffix.lower() in IMG_EXTS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {input_path}")
    return image_paths


def predict_mask(model, image_path: Path, device: torch.device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    resized = image.resize(MODEL_INPUT_SIZE, Image.BILINEAR)
    tensor = transform(resized).unsqueeze(0).to(device)

    with torch.inference_mode():
        predicted = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    mask_image = Image.fromarray(predicted, mode="L").resize(original_size, Image.NEAREST)
    mask = np.array(mask_image, dtype=np.uint8)
    return np.array(image), mask


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[np.clip(mask, 0, NUM_CLASSES - 1)]


def blend_overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    blended = (1.0 - alpha) * image.astype(np.float32) + alpha * color_mask.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)


def save_outputs(image_path: Path, image: np.ndarray, mask: np.ndarray, output_dir: Path, alpha: float):
    masks_dir = output_dir / "masks_single"
    labels_dir = output_dir / "labels_color"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    color_mask = colorize_mask(mask)
    overlay = blend_overlay(image, color_mask, alpha)

    mask_path = masks_dir / f"{stem}_mask.png"
    label_path = labels_dir / f"{stem}_label.png"
    overlay_path = overlays_dir / f"{stem}_overlay.png"

    Image.fromarray(mask, mode="L").save(mask_path)
    Image.fromarray(color_mask).save(label_path)
    Image.fromarray(overlay).save(overlay_path)
    return mask_path, label_path, overlay_path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    image_paths = list_input_images(input_path)
    model, model_name = load_model(weights_path, device, args.model, args.encoder_weights)

    print(f"[device]  {device}")
    print(f"[model]   {model_name}")
    print(f"[weights] {weights_path}")
    print(f"[input]   {len(image_paths)} image(s)")
    print(f"[output]  {output_dir}")

    for image_path in image_paths:
        image, mask = predict_mask(model, image_path, device)
        mask_path, label_path, overlay_path = save_outputs(
            image_path=image_path,
            image=image,
            mask=mask,
            output_dir=output_dir,
            alpha=args.overlay_alpha,
        )
        print(f"\n{image_path.name}")
        print(f"  mask    -> {mask_path}")
        print(f"  label   -> {label_path}")
        print(f"  overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
