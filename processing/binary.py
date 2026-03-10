import numpy as np
from processing.grayscale import apply_grayscale


def apply_binary(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Convert image to binary (black/white) by applying a threshold to grayscale values.
    Pixels above threshold -> 255 (white), below or equal -> 0 (black).
    """
    gray = apply_grayscale(img)
    gray_ch = gray[:, :, 0].astype(np.float32)
    h, w = gray_ch.shape
    output = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            output[i, j] = 255 if gray_ch[i, j] > threshold else 0
    binary_3ch = np.stack([output, output, output], axis=2)
    return binary_3ch
