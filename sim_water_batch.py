"""
Water IR Batch Processor
Reads from images/train and images/val, writes to images_water/train and
images_water/val — preserving exact filenames so YOLO labels stay matched.

Just set PROJECT_DIR below and run:
    python sim_water_batch.py
"""

from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import gaussian_filter


# ── SET THESE ─────────────────────────────────────────────────────────────────
PROJECT_DIR    = Path(".")    # change to your project folder path if needed
WATER_DARKNESS = 0.68
CONTRAST       = 1.45
BASE_SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def detect_people(arr):
    p90  = np.percentile(arr, 90)
    p995 = np.percentile(arr, 99.5)
    top_pixels = arr[arr > p90].flatten()
    hist, edges = np.histogram(top_pixels, bins=60)
    centers = (edges[:-1] + edges[1:]) / 2
    threshold = None
    for i in range(5, len(hist) - 2):
        if hist[i] < hist[i-1] * 0.35 and centers[i] > p90 * 1.05:
            threshold = centers[i]
            break
    if threshold is None:
        threshold = np.percentile(arr, 98.5)
    return float(min(threshold, p995))


def normalize_background(arr, people_threshold):
    bg  = np.where(arr <= people_threshold, arr, np.nan)
    p2  = np.nanpercentile(bg, 2)
    p98 = np.nanpercentile(bg, 98)
    bg_norm   = (np.clip(arr, p2, p98) - p2) / (p98 - p2 + 1e-6) * 135 + 40
    bg_smooth = gaussian_filter(bg_norm, sigma=6)
    return bg_smooth * 0.45 + bg_norm * 0.55


def preserve_people(arr, bg_norm, people_threshold):
    hot_mask = arr > people_threshold
    if not hot_mask.any():
        return bg_norm
    p_lo = people_threshold
    p_hi = np.percentile(arr[hot_mask], 99)
    hot  = (np.clip(arr, p_lo, p_hi) - p_lo) / (p_hi - p_lo + 1e-6) * 40 + 200
    return np.where(hot_mask, hot, bg_norm)


def add_ripple_texture(arr, seed=None):
    rng   = np.random.default_rng(seed)
    noise = rng.normal(0, 1, arr.shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=(2.5, 5.5))
    return arr + noise / (noise.std() + 1e-6) * 3


def add_ir_sensor_noise(arr, seed=None):
    rng      = np.random.default_rng(seed)
    rows, cols = arr.shape
    temporal = rng.normal(0, 2.5, (rows, cols)).astype(np.float32)
    fpn_row  = rng.normal(0, 1.2, (rows, 1)).astype(np.float32)
    fpn_col  = rng.normal(0, 0.8, (1, cols)).astype(np.float32)
    return arr + temporal + fpn_row + fpn_col


def add_specular_reflections(arr, seed=None):
    rng    = np.random.default_rng(seed)
    rows, cols = arr.shape
    result = arr.copy()
    for _ in range(rng.integers(3, 9)):
        r  = rng.integers(5, rows - 5)
        c  = rng.integers(5, cols - 5)
        intensity = rng.uniform(15, 35)
        r0 = max(0, r - int(rng.integers(1, 3))); r1 = min(rows, r + int(rng.integers(1, 3)))
        c0 = max(0, c - int(rng.integers(1, 4))); c1 = min(cols, c + int(rng.integers(1, 4)))
        result[r0:r1, c0:c1] = np.clip(result[r0:r1, c0:c1] + intensity, 0, 255)
    return result


def add_vignette(arr, strength=0.40, seed=None):
    rng    = np.random.default_rng(seed)
    rows, cols = arr.shape
    cx = rng.uniform(0.45, 0.55) * cols
    cy = rng.uniform(0.45, 0.55) * rows
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt(((X - cx) / (cols / 2))**2 + ((Y - cy) / (rows / 2))**2)
    return arr * (1.0 - strength * np.clip(dist, 0, 1) ** 1.5)


def process_image(input_path, output_path, seed):
    rng = np.random.default_rng(seed)
    arr = np.array(Image.open(input_path).convert('L'), dtype=np.float32)

    threshold = detect_people(arr)
    bg_norm   = normalize_background(arr, threshold)
    arr       = preserve_people(arr, bg_norm, threshold)
    arr       = arr * WATER_DARKNESS
    arr       = add_ripple_texture(arr,          seed=rng.integers(1e6))
    arr       = add_ir_sensor_noise(arr,         seed=rng.integers(1e6))
    arr       = add_vignette(arr, strength=0.40, seed=rng.integers(1e6))
    arr       = add_specular_reflections(arr,    seed=rng.integers(1e6))
    arr       = np.clip(arr, 0, 255)

    img_out = ImageEnhance.Contrast(Image.fromarray(arr.astype(np.uint8))).enhance(CONTRAST)
    img_out.save(output_path)


def process_split(split):
    input_dir  = PROJECT_DIR / "images" / split
    output_dir = PROJECT_DIR / "images_water" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in EXTENSIONS])
    if not files:
        print(f"  No images found in {input_dir}, skipping.")
        return

    print(f"\n── {split.upper()} ({len(files)} images) ──")
    print(f"   {input_dir} → {output_dir}\n")

    failed = []
    for i, fpath in enumerate(files):
        out_path = output_dir / fpath.name   # exact same filename = labels stay matched
        try:
            process_image(str(fpath), str(out_path), seed=BASE_SEED + i)
            print(f"  [{i+1:>4}/{len(files)}] {fpath.name}")
        except Exception as e:
            print(f"  [{i+1:>4}/{len(files)}] FAILED {fpath.name}: {e}")
            failed.append(fpath.name)

    print(f"\n  ✓ {len(files) - len(failed)}/{len(files)} saved to {output_dir}/")
    if failed:
        print(f"  ✗ {len(failed)} failed: {failed}")


if __name__ == "__main__":
    print(f"Project dir: {PROJECT_DIR.resolve()}")
    print(f"Output will be written to: images_water/train  and  images_water/val")
    print(f"Labels folder is untouched — point YOLO at images_water/ instead of images/\n")

    for split in ["train", "val"]:
        process_split(split)

    print("\nAll done!")