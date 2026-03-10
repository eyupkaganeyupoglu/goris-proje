import numpy as np


def _make_strel(size: int = 3) -> np.ndarray:
    """Create a default square structuring element of all 1s."""
    return np.ones((size, size), dtype=np.uint8)


def _to_binary(img: np.ndarray) -> np.ndarray:
    """Convert any image to a binary (0 or 1) 2D array for morphological ops."""
    if img.ndim == 3:
        ch = img[:, :, 0]
    else:
        ch = img
    return (ch > 127).astype(np.uint8)


def _from_binary(b: np.ndarray) -> np.ndarray:
    """Convert binary (0/1) array back to 3-channel uint8 (0 or 255)."""
    out = (b * 255).astype(np.uint8)
    return np.stack([out, out, out], axis=2)


def apply_dilate(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    """
    Dilation: if at least one pixel under the structuring element is 1, center -> 1.
    (Local maximum filter)
    """
    binary = _to_binary(img)
    strel = _make_strel(strel_size)
    h, w = binary.shape
    kh, kw = strel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(binary)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = 1 if np.any(region[strel == 1] == 1) else 0
    return _from_binary(output)


def apply_erode(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    """
    Erosion: only if ALL pixels under the structuring element are 1, center -> 1.
    (Local minimum filter)
    """
    binary = _to_binary(img)
    strel = _make_strel(strel_size)
    h, w = binary.shape
    kh, kw = strel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(binary)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = 1 if np.all(region[strel == 1] == 1) else 0
    return _from_binary(output)


def apply_opening(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    """Opening = Erosion then Dilation. Removes small white noise."""
    eroded = apply_erode(img, strel_size)
    return apply_dilate(eroded, strel_size)


def apply_closing(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    """Closing = Dilation then Erosion. Fills small black holes."""
    dilated = apply_dilate(img, strel_size)
    return apply_erode(dilated, strel_size)
