"""
Facial Image Beautification Using Frequency-Domain Techniques
============================================================
This script applies frequency-domain image processing (FFT-based filtering)
to produce "before" (noise-added/degraded) and "after" (beautified) versions
of facial images.

Pipeline per image:
  1. "Before" version : add simulated skin blemishes via high-frequency noise
     injection in the Fourier domain.
  2. "After" version  : apply a cascade of frequency-domain operations:
       a. Gaussian low-pass filter  → skin smoothing
       b. High-pass filter (unsharp masking in frequency domain) → edge/detail sharpening
       c. Band-pass suppression of noise frequencies
       d. CLAHE in LAB color space for balanced contrast / brightness
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def fft2_channel(channel):
    """2-D FFT with shifted DC component."""
    return np.fft.fftshift(np.fft.fft2(channel.astype(np.float64)))

def ifft2_channel(F):
    """Inverse FFT, return real-valued uint8-clipped result."""
    result = np.fft.ifft2(np.fft.ifftshift(F)).real
    return np.clip(result, 0, 255).astype(np.uint8)

def gaussian_lowpass_mask(shape, cutoff):
    """Create a Gaussian low-pass filter mask (0..1)."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    H = np.exp(-(D**2) / (2 * (cutoff**2)))
    return H

def gaussian_highpass_mask(shape, cutoff):
    """Gaussian high-pass = 1 - low-pass."""
    return 1.0 - gaussian_lowpass_mask(shape, cutoff)

def bandstop_mask(shape, low_cut, high_cut):
    """Band-stop (notch) mask: attenuate frequencies between low_cut and high_cut."""
    lp = gaussian_lowpass_mask(shape, low_cut)
    hp = gaussian_highpass_mask(shape, high_cut)
    return lp + hp  # passes low AND high; blocks band in between

def apply_freq_filter(channel, mask):
    """FFT → apply mask → IFFT for a single channel."""
    F = fft2_channel(channel)
    F_filtered = F * mask
    return ifft2_channel(F_filtered)

# ─────────────────────────────────────────────
# "Before" degradation  (simulate blemishes)
# ─────────────────────────────────────────────

def degrade_image(img):
    """
    Simulate a 'before-beautification' face by injecting structured noise
    in the frequency domain (mid-to-high frequency spikes = skin texture noise).
    """
    rows, cols = img.shape[:2]
    shape = (rows, cols)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab.astype(np.float64))

    # Inject random high-frequency noise into L channel in Fourier domain
    FL = fft2_channel(L.astype(np.uint8))

    # Create a noise mask concentrated in mid-high frequency ring
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)

    r_min = min(rows, cols) * 0.15   # inner radius of noise ring
    r_max = min(rows, cols) * 0.40   # outer radius

    noise_region = ((D > r_min) & (D < r_max)).astype(np.float64)
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 600
    FL_noisy = FL + noise * noise_region

    L_noisy = ifft2_channel(FL_noisy).astype(np.float64)
    lab_noisy = cv2.merge([
        np.clip(L_noisy, 0, 255).astype(np.uint8),
        A.astype(np.uint8),
        B.astype(np.uint8)
    ])
    return cv2.cvtColor(lab_noisy, cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────
# "After" beautification pipeline
# ─────────────────────────────────────────────

def beautify_image(img):
    """
    Frequency-domain beautification pipeline:
      1. Gaussian LPF on L channel → skin smoothing
      2. Unsharp masking via HPF → edge sharpening
      3. Band-stop on noise ring  → noise removal
      4. CLAHE                    → contrast / brightness balance
    """
    rows, cols = img.shape[:2]
    shape = (rows, cols)

    # Work in LAB: process luminance (L) separately from colour (A,B)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # ── Step 1 : Gaussian Low-Pass → smooth skin texture ──────────────────
    lp_cutoff = min(rows, cols) * 0.12          # keep only low-freq content
    lp_mask   = gaussian_lowpass_mask(shape, lp_cutoff)
    L_smooth  = apply_freq_filter(L, lp_mask).astype(np.float64)

    # ── Step 2 : Unsharp masking (frequency-domain) → sharpen features ────
    hp_cutoff  = min(rows, cols) * 0.08
    hp_mask    = gaussian_highpass_mask(shape, hp_cutoff)
    L_edges    = apply_freq_filter(L, hp_mask).astype(np.float64)
    sharp_amt  = 1.4                            # boosting factor for edges
    L_sharp    = np.clip(L_smooth + sharp_amt * L_edges, 0, 255)

    # ── Step 3 : Band-stop (suppress mid-frequency noise ring) ─────────────
    low_cut  = min(rows, cols) * 0.10
    high_cut = min(rows, cols) * 0.38
    bs_mask  = bandstop_mask(shape, low_cut, high_cut)
    L_clean  = apply_freq_filter(L_sharp.astype(np.uint8), bs_mask).astype(np.float64)

    # ── Step 4 : Mild smoothing blend to keep skin natural ─────────────────
    lp_fine   = gaussian_lowpass_mask(shape, min(rows, cols) * 0.25)
    L_fine    = apply_freq_filter(L_clean.astype(np.uint8), lp_fine).astype(np.float64)
    L_final   = np.clip(0.65 * L_clean + 0.35 * L_fine, 0, 255).astype(np.uint8)

    # Reconstruct LAB image
    lab_out = cv2.merge([L_final, A, B])
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)

    # ── Step 5 : CLAHE for balanced contrast / brightness ──────────────────
    lab2 = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2LAB)
    L2, A2, B2 = cv2.split(lab2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2_eq = clahe.apply(L2)
    lab_clahe = cv2.merge([L2_eq, A2, B2])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return result

# ─────────────────────────────────────────────
# Visualise frequency spectrum (for report)
# ─────────────────────────────────────────────

def magnitude_spectrum(channel):
    F = np.fft.fftshift(np.fft.fft2(channel))
    mag = 20 * np.log1p(np.abs(F))
    return (mag / mag.max() * 255).astype(np.uint8)

# ─────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────

def process_face(input_path, label, out_dir):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {input_path}")

    # Resize to a manageable size for uniform processing
    h, w = img.shape[:2]
    max_dim = 512
    scale = min(max_dim / h, max_dim / w, 1.0)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    before = degrade_image(img)
    after  = beautify_image(before)   # beautify the degraded version

    # Save images
    orig_path   = os.path.join(out_dir, f"{label}_original.jpg")
    before_path = os.path.join(out_dir, f"{label}_before.jpg")
    after_path  = os.path.join(out_dir, f"{label}_after.jpg")

    cv2.imwrite(orig_path,   img)
    cv2.imwrite(before_path, before)
    cv2.imwrite(after_path,  after)

    # ── Comparison figure ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, title in zip(
        axes,
        [img, before, after],
        ["Original", "Before Beautification\n(freq-domain noise added)", "After Beautification\n(freq-domain filtering)"]
    ):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axis('off')
    plt.suptitle(f"Image: {label}", fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{label}_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ── Spectrum figure ────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    for ax, im, title in zip(
        axes2,
        [before, after],
        ["Spectrum – Before", "Spectrum – After"]
    ):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        spec = magnitude_spectrum(gray)
        ax.imshow(spec, cmap='inferno')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.suptitle(f"Frequency Spectrum – {label}", fontsize=13)
    plt.tight_layout()
    spec_path = os.path.join(out_dir, f"{label}_spectrum.png")
    plt.savefig(spec_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[{label}] Saved: {before_path}, {after_path}, {fig_path}, {spec_path}")
    return orig_path, before_path, after_path, fig_path, spec_path


if __name__ == "__main__":
    OUT = "/home/claude/output"
    os.makedirs(OUT, exist_ok=True)

    results = {}
    for label, src in [("face1", "/tmp/face1.jpg"), ("face2", "/tmp/face2.jpg")]:
        results[label] = process_face(src, label, OUT)

    print("\nAll done. Output directory:", OUT)
    import subprocess
    subprocess.run(["ls", "-lh", OUT])
