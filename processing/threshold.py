import numpy as np
from processing.grayscale import apply_grayscale

# Eşikleme (thresholding) işlemini uygular
def apply_threshold(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    # Görüntüyü gri tonlamaya çevirir ve boyutlarını belirler
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    
    # Sonuç için boş bir matris oluşturur
    output = np.zeros((h, w), dtype=np.uint8)
    
    # Her bir pikseli belirlenen eşik değeriyle karşılaştırıp siyah veya beyaz yapar
    for i in range(h):
        for j in range(w):
            output[i, j] = 255 if int(channel[i, j]) > threshold else 0
            
    # Tek kanallı sonucu tekrar 3 kanallı (RGB) yapıya getirir
    return np.stack([output, output, output], axis=2)
