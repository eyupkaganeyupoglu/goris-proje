import numpy as np

def apply_brightness_multiply(img: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Parlaklığı artırmak için her pikseli alfa ile çarpın.
    g(x,y) = alfa * f(x,y). Değerler 0-255 aralığına kırpılır.
    """
    result = img.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_contrast_adjust(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Orta nokta (128) etrafında kontrastı ayarlar.
    factor > 1: kontrastı artırır, factor < 1: kontrastı azaltır.
    Q(i,j) = factor * (P(i,j) - 128) + 128
    """
    img_f = img.astype(np.float32)
    result = factor * (img_f - 128.0) + 128.0
    return np.clip(result, 0, 255).astype(np.uint8)