import numpy as np
from processing.zoom import apply_zoom

def _match_dimensions(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img1.shape != img2.shape:
        scale_h = img1.shape[0] / img2.shape[0]
        scale_w = img1.shape[1] / img2.shape[1]
        scale = max(scale_h, scale_w)
        img2 = apply_zoom(img2, scale)
        img2 = img2[:img1.shape[0], :img1.shape[1]]
    return img2

def add_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    img2 = _match_dimensions(img1, img2)
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    result = (f1 + f2) / 2.0
    return np.clip(result, 0, 255).astype(np.uint8)

def divide_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    img2 = _match_dimensions(img1, img2)
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    safe_f2 = np.where(f2 == 0, 1e-6, f2)
    result = f1 / safe_f2
    return np.clip(result, 0, 255).astype(np.uint8)