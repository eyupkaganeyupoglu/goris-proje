import numpy as np


def add_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    result = (f1 + f2) / 2.0
    return np.clip(result, 0, 255).astype(np.uint8)


def divide_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    safe_f2 = np.where(f2 == 0, 1e-6, f2)
    result = f1 / safe_f2
    return np.clip(result, 0, 255).astype(np.uint8)
