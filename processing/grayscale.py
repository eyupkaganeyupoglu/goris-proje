import numpy as np

def apply_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Gri Yoğunluk = 0.299*R + 0.587*G + 0.114*B

    'Grayscale' tuşuna basınca direkt çalışıyor.
    """
    img_f = img.astype(np.float32)
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    gray_3ch = np.stack([Y, Y, Y], axis=2)
    return gray_3ch