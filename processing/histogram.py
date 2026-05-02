import numpy as np
from processing.grayscale import apply_grayscale

def compute_histogram(img: np.ndarray) -> list:
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(channel[i, j])] += 1
    return hist

def apply_histogram_stretch(img: np.ndarray, a: int = 0, b: int = 255) -> np.ndarray:
    img_f = img.astype(np.float32)
    c = float(img_f.min())
    d = float(img_f.max())
    if d == c:
        return img
    stretched = (img_f - c) * ((b - a) / (d - c)) + a
    return np.clip(stretched, 0, 255).astype(np.uint8)

def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    total_pixels = h * w

    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(channel[i, j])] += 1

    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = 0
    for i in range(256):
        if cdf[i] != 0:
            cdf_min = cdf[i]
            break

    lookup = [0] * 256
    denom = total_pixels - cdf_min
    if denom == 0:
        return gray
    for i in range(256):
        lookup[i] = int(round((cdf[i] - cdf_min) / denom * 255))

    output = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            val = lookup[int(channel[i, j])]
            output[i, j] = [val, val, val]

    return output.astype(np.uint8)

def apply_histogram_expand(img: np.ndarray, a: float = 0.3, b: float = 0.7) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0

    lo = float(a)
    hi = float(b)
    if hi <= lo:
        return img
    expanded = (1.0 / (hi - lo)) * (img_f - lo)
    expanded = np.clip(expanded, 0.0, 1.0)
    return (expanded * 255.0).astype(np.uint8)