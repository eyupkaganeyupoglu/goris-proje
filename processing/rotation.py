import math
import numpy as np


def apply_rotation(img: np.ndarray, angle_deg: float = 45.0) -> np.ndarray:
    """
    Rotate image around its own center using inverse transformation + nearest-neighbor interpolation.
    No cv2.rotate or cv2.warpAffine used.
    """
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    channels = img.shape[2] if img.ndim == 3 else 1
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            # Inverse rotation: map destination -> source
            dy = i - cy
            dx = j - cx
            src_x = cos_a * dx + sin_a * dy + cx
            src_y = -sin_a * dx + cos_a * dy + cy
            si = int(round(src_y))
            sj = int(round(src_x))
            if 0 <= si < h and 0 <= sj < w:
                output[i, j] = img[si, sj]
    return output
