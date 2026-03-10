import numpy as np


def apply_yuv_conversion(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to YCrCb (YUV) manually using the formula from project-details.md.
    Y  =  0.257*R + 0.504*G + 0.098*B + 16
    Cr =  0.439*R - 0.368*G - 0.071*B + 128
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Returns a 3-channel uint8 image with [Y, Cr, Cb] channels.
    """
    img_f = img.astype(np.float32)
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]

    Y  =  0.257 * R + 0.504 * G + 0.098 * B + 16
    Cr =  0.439 * R - 0.368 * G - 0.071 * B + 128
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128

    Y  = np.clip(Y,  0, 255)
    Cr = np.clip(Cr, 0, 255)
    Cb = np.clip(Cb, 0, 255)

    yuv = np.stack([Y, Cr, Cb], axis=2).astype(np.uint8)
    return yuv
