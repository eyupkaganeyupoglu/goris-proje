import numpy as np


def apply_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Crop a region from the image using coordinate slicing.
    (x1, y1) = top-left corner, (x2, y2) = bottom-right corner.
    """
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return img[y1:y2, x1:x2]
