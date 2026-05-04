import numpy as np
from processing.binary import apply_binary

# Kare şeklinde bir structuring element (strel) oluşturur
def _make_strel(size: int = 3) -> np.ndarray:
    return np.ones((size, size), dtype=np.uint8)

# Görüntüyü [0, 1] değerlerinden oluşan binary formata çevirir
def _to_binary(img: np.ndarray) -> np.ndarray:
    binary_3ch = apply_binary(img, threshold=127)
    return (binary_3ch[:, :, 0] // 255).astype(np.uint8)

# Binary görüntüyü tekrar 3 kanallı (RGB) formatına getirir
def _from_binary(b: np.ndarray) -> np.ndarray:
    out = (b * 255).astype(np.uint8)
    return np.stack([out, out, out], axis=2)

# Yayma (dilation) işlemini uygular
def apply_dilation(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    # Görüntüyü ve structuring element'i hazırlar
    binary = _to_binary(img)
    strel = _make_strel(strel_size)
    h, w = binary.shape
    kh, kw = strel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(binary)
    
    # Komşuluk içinde herhangi bir piksel '1' ise hedefi '1' yapar
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = 1 if np.any(region[strel == 1] == 1) else 0
    return _from_binary(output)

# Aşındırma (erosion) işlemini uygular
def apply_erosion(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    # Görüntüyü ve structuring element'i hazırlar
    binary = _to_binary(img)
    strel = _make_strel(strel_size)
    h, w = binary.shape
    kh, kw = strel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(binary, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(binary)
    
    # Komşuluktaki tüm pikseller '1' ise hedefi '1' yapar
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = 1 if np.all(region[strel == 1] == 1) else 0
    return _from_binary(output)

# Açma (opening) işlemi: Önce erosion, sonra dilation
def apply_opening(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    eroded = apply_erosion(img, strel_size)
    return apply_dilation(eroded, strel_size)

# Kapama (closing) işlemi: Önce dilation, sonra erosion
def apply_closing(img: np.ndarray, strel_size: int = 3) -> np.ndarray:
    dilated = apply_dilation(img, strel_size)
    return apply_erosion(dilated, strel_size)