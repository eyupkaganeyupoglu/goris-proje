import numpy as np
from processing.binary import apply_binary

def _make_strel(size: int = 3) -> np.ndarray:
    return np.ones((size, size), dtype=np.uint8)

def _to_binary(img: np.ndarray) -> np.ndarray:
    binary_3ch = apply_binary(img, threshold=127)
    return (binary_3ch[:, :, 0] // 255).astype(np.uint8)

def _from_binary(b: np.ndarray) -> np.ndarray:
    out = (b * 255).astype(np.uint8)
    return np.stack([out, out, out], axis=2)

def apply_dilation(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
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

def apply_erosion(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
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
    eroded = apply_erosion(img, strel_size)
    return apply_dilation(eroded, strel_size)

def apply_closing(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    dilated = apply_dilation(img, strel_size)
    return apply_erosion(dilated, strel_size)