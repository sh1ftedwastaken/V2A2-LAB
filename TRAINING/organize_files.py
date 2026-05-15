"""
Run this ONCE to organize your messy folder into:
  data/images/   ← all .jpg files
  data/masks/    ← all .png files
  data/jsons/    ← all .json files
"""
import os, shutil

SOURCE_FOLDER = "old_data"   # <-- change this to your folder path if needed

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks",  exist_ok=True)
os.makedirs("data/jsons",  exist_ok=True)

moved = {"jpg": 0, "png": 0, "json": 0}

for filename in os.listdir(SOURCE_FOLDER):
    src = os.path.join(SOURCE_FOLDER, filename)
    if not os.path.isfile(src):
        continue
    if filename.endswith(".png"):
        shutil.move(src, os.path.join("data/images", filename))
        moved["png"] += 1
    elif filename.endswith(".jpg"):
        shutil.move(src, os.path.join("data/masks", filename))
        moved["jpg"] += 1
    elif filename.endswith(".json"):
        shutil.move(src, os.path.join("data/jsons", filename))
        moved["json"] += 1

print(f"Done! Copied: {moved['png']} images, {moved['jpg']} masks, {moved['json']} JSONs")
