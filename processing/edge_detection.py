import numpy as np
from processing.grayscale import apply_grayscale

# Prewitt kenar tespit operatörleri (Yatay ve Dikey)
GX = np.array([[-1, 0, 1],
               [-1, 0, 1],
               [-1, 0, 1]], dtype=np.float32)

GY = np.array([[-1, -1, -1],
               [ 0,  0,  0],
               [ 1,  1,  1]], dtype=np.float32)

# Prewitt yöntemiyle kenar tespiti yapar
def apply_edge_prewitt(img: np.ndarray) -> np.ndarray:
    # Görüntüyü griye çevirir ve tek kanal üzerinden işlem yapar
    gray = apply_grayscale(img)
    channel = gray[:, :, 0].astype(np.float32)

    # Kenar piksellerini korumak için padding uygular
    h, w = channel.shape
    kh, kw = GX.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Gradyan sonuçları için boş matrisler hazırlar
    result_x = np.zeros((h, w), dtype=np.float32)
    result_y = np.zeros((h, w), dtype=np.float32)

    # Yatay ve dikey yöndeki kenar şiddetlerini hesaplar
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            result_x[i, j] = np.sum(region * GX)
            result_y[i, j] = np.sum(region * GY)

    # Gradyanların mutlak değerlerini toplayarak toplam kenar şiddetini bulur
    G = np.abs(result_x) + np.abs(result_y)
    G = np.clip(G, 0, 255).astype(np.uint8)
    
    # Sonucu tekrar 3 kanallı yapıya getirir
    return np.stack([G, G, G], axis=2)