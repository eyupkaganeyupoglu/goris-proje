import numpy as np
import math


def apply_contrast_multiply(img: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Kontrastı artırmak için her pikseli alfa ile çarpın.
    g(x,y) = alfa * f(x,y). Değerler 0-255 aralığına kırpılır.
    """
    result = img.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast_log(img: np.ndarray) -> np.ndarray:
    """
    Karanlık bölgeleri iyileştirmek için logaritmik dönüşüm uygulayın.
    Q(i,j) = c * log(1 + |P(i,j)|)
    c = 255 / log(1 + maksimum_piksel_değeri)
    """
    img_f = img.astype(np.float32)
    max_val = float(img_f.max())
    if max_val == 0:
        return img
    c = 255.0 / math.log(1.0 + max_val)
    result = c * np.log(1.0 + np.abs(img_f))
    return np.clip(result, 0, 255).astype(np.uint8)
