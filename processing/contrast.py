import numpy as np
import math


def apply_contrast_multiply(img: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Multiply each pixel by alpha to enhance contrast.
    g(x,y) = alpha * f(x,y). Values clipped to 0-255.
    """
    result = img.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast_log(img: np.ndarray) -> np.ndarray:
    """
    Apply logarithmic transformation to enhance dark regions.
    Q(i,j) = c * log(1 + |P(i,j)|)
    c = 255 / log(1 + max_pixel_value)
    """
    img_f = img.astype(np.float32)
    max_val = float(img_f.max())
    if max_val == 0:
        return img
    c = 255.0 / math.log(1.0 + max_val)
    result = c * np.log(1.0 + np.abs(img_f))
    return np.clip(result, 0, 255).astype(np.uint8)
