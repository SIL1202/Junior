import glob, os

imgs = sorted(glob.glob("train/images/*.*"))
missing = []
for img in imgs:
    base = os.path.splitext(os.path.basename(img))[0]
    if not os.path.exists(f"train/labels/{base}.txt"):
        missing.append(base)
print(
    "[OK] all matched"
    if not missing
    else f"[MISS] {len(missing)} labels missing: {missing[:10]}"
)
