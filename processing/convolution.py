import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2D convolution of a single-channel image with a kernel.
    Uses edge-padding to preserve image dimensions.
    Q(i,j) = sum_k sum_l [ K(k,l) * P(i-k, j-l) ]
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_mean_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Apply a mean (average) filter using a size x size kernel of all 1/(size*size).
    Works channel by channel for color images.
    """
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    if img.ndim == 2:
        return convolve(img, kernel)
    channels = [convolve(img[:, :, c], kernel) for c in range(img.shape[2])]
    return np.stack(channels, axis=2)
