"""
Run this ONCE to organize your messy folder into:
  data/images/   ← all .jpg files
  data/masks/    ← all .png files
  data/jsons/    ← all .json files
"""
import os, shutil

SOURCE_FOLDER = "C:\\Users\\zouda\\Downloads\\OneDrive_1_18-08-2026\\clockwise_innerlane_limo_frames"   # <-- change this to your folder path if needed

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks",  exist_ok=True)
os.makedirs("data/jsons",  exist_ok=True)

moved = {"jpg": 0, "png": 0, "json": 0}

for filename in os.listdir(SOURCE_FOLDER):
    src = os.path.join(SOURCE_FOLDER, filename)
    if not os.path.isfile(src):
        continue
    
    if filename.endswith(".jpg"):
        filename = filename[:-4]
        png = filename + ".png"
        json = filename + ".json"
        jpg = filename + ".jpg"
        shutil.move(os.path.join(SOURCE_FOLDER, jpg), os.path.join("data/masks", jpg))
        moved["jpg"] += 1
        shutil.move(os.path.join(SOURCE_FOLDER, png), os.path.join("data/images", png))
        moved["png"] += 1
        shutil.move(os.path.join(SOURCE_FOLDER, json), os.path.join("data/jsons", json))
        moved["json"] += 1
        
print("Organizing files...")
print(f"Done! Moved: {moved['png']} images, {moved['jpg']} masks, {moved['json']} JSONs")
