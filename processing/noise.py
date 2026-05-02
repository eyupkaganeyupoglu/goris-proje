import numpy as np

def add_salt_pepper(img: np.ndarray, amount: float = 0.05) -> np.ndarray:
    result = img.copy()
    h, w = img.shape[:2]
    total = h * w
    n_noise = int(total * amount)

    rng = np.random.default_rng()

    coords_salt = rng.integers(0, [h, w], size=(n_noise // 2, 2))
    for y, x in coords_salt:
        result[y, x] = 255

    coords_pepper = rng.integers(0, [h, w], size=(n_noise // 2, 2))
    for y, x in coords_pepper:
        result[y, x] = 0

    return result

def clean_mean(img: np.ndarray) -> np.ndarray:
    from processing.convolution import apply_mean_filter
    return apply_mean_filter(img, size=3)

def clean_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    pad = size // 2
    result = np.zeros_like(img)

    if img.ndim == 3:
        h, w, c = img.shape
        for ch in range(c):
            channel = img[:, :, ch].astype(np.uint8)
            padded = np.pad(channel, pad, mode='edge')
            out_ch = np.zeros((h, w), dtype=np.uint8)
            for i in range(h):
                for j in range(w):
                    window = padded[i:i + size, j:j + size].flatten().tolist()
                    window.sort()
                    out_ch[i, j] = window[len(window) // 2]
            result[:, :, ch] = out_ch
    else:
        h, w = img.shape
        padded = np.pad(img, pad, mode='edge')
        for i in range(h):
            for j in range(w):
                window = padded[i:i + size, j:j + size].flatten().tolist()
                window.sort()
                result[i, j] = window[len(window) // 2]
    return result