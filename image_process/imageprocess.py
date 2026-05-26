import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# 1. Image Preparation (Homework 1.1)
# ==========================================
# Load the original image in grayscale
img = cv2.imread("moon.jpeg", cv2.IMREAD_GRAYSCALE)

# (a) Dark: Scale down intensity
dark_img = (img * 0.3).astype(np.uint8)

# (b) Bright: Increase intensity and clip at 255
bright_img = np.clip(img.astype(np.float32) + 100, 0, 255).astype(np.uint8)

# (c) Low Contrast: Compress range to 100-150
low_contrast = (img * (50 / 255) + 100).astype(np.uint8)

# (d) High Contrast: Stretch range 50-200 to 0-255
high_contrast = np.clip((img.astype(np.float32) - 50) * (255 / 150), 0, 255).astype(
    np.uint8
)

test_images = [dark_img, bright_img, low_contrast, high_contrast]
titles = ["Dark", "Bright", "Low Contrast", "High Contrast"]

# ==========================================
# 2. Custom Processing Functions (Homework 1.1b)
# ==========================================


def my_histogram_equalization(input_img):
    """Manually implement Histogram Equalization."""
    # Step 1: Calculate Histogram
    hist, _ = np.histogram(input_img.flatten(), 256, [0, 256])

    # Step 2: Calculate CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()

    # Step 3: Normalize CDF to [0, 255]
    cdf_m = np.ma.masked_equal(cdf, 0)  # Mask zeros to handle empty bins
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype("uint8")

    # Step 4: Map original pixels using CDF as lookup table
    output_img = cdf_final[input_img]
    return output_img, cdf_final


def power_law_transformation(input_img, gamma=0.5):
    """Implement Power Law (Gamma Correction)."""
    # Normalize to [0, 1], apply gamma, then scale back to [255]
    res = np.array(255 * (input_img / 255) ** gamma, dtype="uint8")
    return res


# ==========================================
# 3. Processing Results
# ==========================================
he_results = []
transfer_curves = []

for t_img in test_images:
    res, curve = my_histogram_equalization(t_img)
    he_results.append(res)
    transfer_curves.append(curve)

# Example of Power Law for Dark Image Comparison (Homework 1.1b discussion)
power_law_dark = power_law_transformation(dark_img, gamma=0.4)

# ==========================================
# 4. Visualization for Report (Homework 1.1a & 1.1b)
# ==========================================
fig, axes = plt.subplots(5, 4, figsize=(20, 20))

for i in range(4):
    # Row 1: Original Test Images
    # axes[0, i].imshow(test_images[i], cmap="gray", vmin=0, vmax=255)
    # axes[0, i].set_title(f"Original: {titles[i]}")
    # axes[0, i].axis("off")

    # Row 2: Original Histograms
    # axes[0, i].hist(test_images[i].ravel(), 256, [0, 256], color="black")
    # axes[0, i].set_title(f"Original Hist: {titles[i]}")

    # Row 3: HE Result Images
    # axes[0, i].imshow(he_results[i], cmap="gray", vmin=0, vmax=255)
    # axes[0, i].set_title(f"HE Result: {titles[i]}")
    # axes[0, i].axis("off")
    #
    # # Row 4: HE Result Histograms
    # axes[1, i].hist(he_results[i].ravel(), 256, [0, 256], color="blue")
    # axes[1, i].set_title(f"HE Hist: {titles[i]}")

    # Row 5: Transfer Curves (CDF)
    axes[0, i].plot(transfer_curves[i], color="red", linewidth=2)
    axes[0, i].set_title(f"Transfer Curve (CDF): {titles[i]}")
    axes[0, i].set_xlim([0, 255])
    axes[0, i].set_ylim([0, 255])
    axes[0, i].grid(True)

plt.tight_layout()
plt.show()

# Save one comparison for discussion
cv2.imwrite("Comparison_HE_vs_PowerLaw.png", np.hstack((he_results[0], power_law_dark)))
