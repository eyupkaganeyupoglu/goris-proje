import numpy as np


def apply_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale using the YUV luminance formula.
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Returns a single-channel uint8 image (saved as 3-channel for browser compat).
    """
    img_f = img.astype(np.float32)
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    # Stack to 3-channel so cv2.imwrite works correctly in all cases
    gray_3ch = np.stack([Y, Y, Y], axis=2)
    return gray_3ch
