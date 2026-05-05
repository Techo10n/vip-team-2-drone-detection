from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import gaussian_filter

# ── TUNING KNOBS ─────────────────────────────────────────────────────────────
INPUT_IMAGE = "test_img.jpg"  # path to your input image
OUTPUT_IMAGE = "sim_water_out_NEWDAY.png"

WATER_DARKNESS = 0.68  # overall water darkness (lower = darker)
CONTRAST = 1.45  # final contrast boost
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────


def detect_people(arr):
    """
    Adaptively find the threshold separating hot objects (people/vehicles)
    from background by looking for a histogram valley above p90.
    Falls back to p98.5 if no clean valley found.
    """
    p90 = np.percentile(arr, 90)
    p995 = np.percentile(arr, 99.5)

    top_pixels = arr[arr > p90].flatten()
    hist, edges = np.histogram(top_pixels, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2

    threshold = None
    for i in range(5, len(hist) - 2):
        if hist[i] < hist[i - 1] * 0.35 and centers[i] > p90 * 1.05:
            threshold = centers[i]
            break

    if threshold is None:
        threshold = np.percentile(arr, 98.5)

    return float(min(threshold, p995))


def normalize_background(arr, people_threshold):
    """
    Remap background pixels to a consistent 40-175 range regardless of
    source surface (courts, grass, roads, etc.), with light smoothing
    to kill surface texture.
    """
    bg = np.where(arr <= people_threshold, arr, np.nan)
    p2 = np.nanpercentile(bg, 2)
    p98 = np.nanpercentile(bg, 98)

    bg_norm = np.clip(arr, p2, p98)
    bg_norm = (bg_norm - p2) / (p98 - p2 + 1e-6) * 135 + 40

    bg_smooth = gaussian_filter(bg_norm, sigma=6)
    bg_norm = bg_smooth * 0.45 + bg_norm * 0.55

    return bg_norm


def preserve_people(arr, bg_norm, people_threshold):
    """
    Remap hot objects to 200-240 DN — clearly visible but not blown out.
    """
    hot_mask = arr > people_threshold
    if not hot_mask.any():
        return bg_norm

    p_lo = people_threshold
    p_hi = np.percentile(arr[hot_mask], 99)
    hot = (np.clip(arr, p_lo, p_hi) - p_lo) / (p_hi - p_lo + 1e-6) * 40 + 200

    return np.where(hot_mask, hot, bg_norm)


def add_ripple_texture(arr, seed=None):
    """Fine water surface texture — very subtle (~3 DN std)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, arr.shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=(2.5, 5.5))
    noise = noise / (noise.std() + 1e-6) * 3
    return arr + noise


def add_ir_sensor_noise(arr, seed=None):
    """FLIR-style fixed-pattern noise + temporal read noise."""
    rng = np.random.default_rng(seed)
    rows, cols = arr.shape
    temporal = rng.normal(0, 2.5, (rows, cols)).astype(np.float32)
    fpn_row = rng.normal(0, 1.2, (rows, 1)).astype(np.float32)
    fpn_col = rng.normal(0, 0.8, (1, cols)).astype(np.float32)
    return arr + temporal + fpn_row + fpn_col


def add_specular_reflections(arr, seed=None):
    """Tiny sparse IR glints on water surface."""
    rng = np.random.default_rng(seed)
    rows, cols = arr.shape
    result = arr.copy()
    for _ in range(rng.integers(3, 9)):
        r = rng.integers(5, rows - 5)
        c = rng.integers(5, cols - 5)
        intensity = rng.uniform(15, 35)
        r0 = max(0, r - int(rng.integers(1, 3)))
        r1 = min(rows, r + int(rng.integers(1, 3)))
        c0 = max(0, c - int(rng.integers(1, 4)))
        c1 = min(cols, c + int(rng.integers(1, 4)))
        result[r0:r1, c0:c1] = np.clip(result[r0:r1, c0:c1] + intensity, 0, 255)
    return result


def add_vignette(arr, strength=0.40, seed=None):
    """Darken edges to simulate lens vignetting."""
    rng = np.random.default_rng(seed)
    rows, cols = arr.shape
    cx = rng.uniform(0.45, 0.55) * cols
    cy = rng.uniform(0.45, 0.55) * rows
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt(((X - cx) / (cols / 2)) ** 2 + ((Y - cy) / (rows / 2)) ** 2)
    vignette = 1.0 - strength * np.clip(dist, 0, 1) ** 1.5
    return arr * vignette


# ── PIPELINE ──────────────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED)

print(f"Loading {INPUT_IMAGE}...")
arr = np.array(Image.open(INPUT_IMAGE).convert('L'), dtype=np.float32)

print("Detecting people/hot objects...")
threshold = detect_people(arr)
print(f"  Threshold: {threshold:.1f} DN  ({(arr > threshold).mean() * 100:.2f}% of pixels)")

print("Normalizing background...")
bg_norm = normalize_background(arr, threshold)

print("Preserving people...")
arr = preserve_people(arr, bg_norm, threshold)

print("Applying water darkness...")
arr = arr * WATER_DARKNESS

print("Adding ripple texture...")
arr = add_ripple_texture(arr, seed=rng.integers(1e6))

print("Adding sensor noise...")
arr = add_ir_sensor_noise(arr, seed=rng.integers(1e6))

print("Adding vignette...")
arr = add_vignette(arr, strength=0.40, seed=rng.integers(1e6))

print("Adding specular glints...")
arr = add_specular_reflections(arr, seed=rng.integers(1e6))

arr = np.clip(arr, 0, 255)
img_out = ImageEnhance.Contrast(Image.fromarray(arr.astype(np.uint8))).enhance(CONTRAST)
img_out.save(OUTPUT_IMAGE)

out = np.array(img_out)
print(f"\nMean={out.mean():.1f}  Std={out.std():.1f}  "
      f"People(>190)={(out > 190).mean() * 100:.2f}%")
print(f"Saved → {OUTPUT_IMAGE}")