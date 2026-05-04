import numpy as np
from processing.convolution import convolve

# Keskinleştirme için kullanılan Laplacian kernel matrisi
UNSHARP_KERNEL = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float32)

# Unsharp filtresini uygular
def apply_unsharp(img: np.ndarray) -> np.ndarray:
    # Görüntü tek kanallıysa doğrudan convolution uygular
    if img.ndim == 2:
        return convolve(img, UNSHARP_KERNEL)
        
    # Çok kanallı görüntülerde her kanal için ayrı ayrı keskinleştirme yapar
    channels = [convolve(img[:, :, c], UNSHARP_KERNEL) for c in range(img.shape[2])]
    return np.stack(channels, axis=2)
