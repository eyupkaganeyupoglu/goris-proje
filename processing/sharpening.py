import numpy as np
from processing.convolution import convolve

UNSHARP_KERNEL = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]], dtype=np.float32)

def apply_unsharp(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return convolve(img, UNSHARP_KERNEL)
    channels = [convolve(img[:, :, c], UNSHARP_KERNEL) for c in range(img.shape[2])]
    return np.stack(channels, axis=2)
