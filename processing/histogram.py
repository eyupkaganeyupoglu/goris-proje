import numpy as np
from processing.grayscale import apply_grayscale


def compute_histogram(img: np.ndarray) -> list:
    """
    Manually compute the frequency of each pixel intensity (0-255) by looping.
    Returns a 256-element list where index = intensity, value = count.
    """
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(channel[i, j])] += 1
    return hist


def apply_histogram_stretch(img: np.ndarray, a: int = 0, b: int = 255) -> np.ndarray:
    """
    Apply linear histogram stretching.
    New_Pixel = (Old_Pixel - c) * ((b - a) / (d - c)) + a
    where c = min value in image, d = max value in image.
    """
    img_f = img.astype(np.float32)
    c = float(img_f.min())
    d = float(img_f.max())
    if d == c:
        return img  # avoid division by zero — all pixels same value
    stretched = (img_f - c) * ((b - a) / (d - c)) + a
    return np.clip(stretched, 0, 255).astype(np.uint8)
