# make_val_split.py
import os, random, shutil, glob, pathlib

random.seed(42)

assert os.path.isdir("train/images") and os.path.isdir(
    "train/labels"
), "failed to find train/images or train/labels"

imgs = sorted(glob.glob("train/images/*.*"))
k = max(1, len(imgs) // 5)  # 20%
pick = set(random.sample(imgs, k))

pathlib.Path("val/images").mkdir(parents=True, exist_ok=True)
pathlib.Path("val/labels").mkdir(parents=True, exist_ok=True)

moved = 0
for img in pick:
    base = os.path.splitext(os.path.basename(img))[0]
    lbl = f"train/labels/{base}.txt"
    shutil.move(img, f"val/images/{os.path.basename(img)}")
    if os.path.exists(lbl):
        shutil.move(lbl, f"val/labels/{os.path.basename(lbl)}")
        moved += 1

print(f"[OK] moved {moved} pairs to val/")
