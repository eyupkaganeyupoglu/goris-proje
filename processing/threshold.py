import numpy as np
from processing.grayscale import apply_grayscale


def apply_threshold(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Single thresholding: scan each pixel of the grayscale image.
    f(x,y) > threshold -> 255 (white), otherwise -> 0 (black).
    """
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    output = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            output[i, j] = 255 if int(channel[i, j]) > threshold else 0
    return np.stack([output, output, output], axis=2)
