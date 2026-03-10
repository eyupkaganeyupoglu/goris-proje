import numpy as np
from processing.convolution import convolve
from processing.grayscale import apply_grayscale


# Prewitt kernels
GX = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]], dtype=np.float32)

GY = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]], dtype=np.float32)


def apply_edge_prewitt(img: np.ndarray) -> np.ndarray:
    """
    Prewitt edge detection using manual convolution with Gx and Gy kernels.
    G = |Gx| + |Gy|  (sum of absolute values)
    Applied on grayscale version of the image.
    """
    gray = apply_grayscale(img)
    channel = gray[:, :, 0].astype(np.float32)

    h, w = channel.shape
    kh, kw = GX.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    result_x = np.zeros((h, w), dtype=np.float32)
    result_y = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            result_x[i, j] = np.sum(region * GX)
            result_y[i, j] = np.sum(region * GY)

    G = np.abs(result_x) + np.abs(result_y)
    G = np.clip(G, 0, 255).astype(np.uint8)
    return np.stack([G, G, G], axis=2)
