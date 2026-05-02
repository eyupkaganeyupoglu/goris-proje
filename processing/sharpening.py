import numpy as np
from processing.convolution import convolve

SHARPEN_KERNEL = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]], dtype=np.float32)

def apply_sharpening(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return convolve(img, SHARPEN_KERNEL)
    channels = [convolve(img[:, :, c], SHARPEN_KERNEL) for c in range(img.shape[2])]
    return np.stack(channels, axis=2)
