import numpy as np
from processing.grayscale import apply_grayscale

def apply_binary(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    # Görüntüyü eşikleme işleminden önce gri tonlamaya dönüştürür
    gray = apply_grayscale(img)
    
    # Tek kanalı alır ve görüntünün boyutlarını belirler
    gray_ch = gray[:, :, 0].astype(np.float32)
    h, w = gray_ch.shape
    
    # Sonuç için aynı boyutlarda boş bir matris oluşturur
    output = np.zeros((h, w), dtype=np.uint8)
    
    # Her pikseli threshold değeriyle karşılaştırıp siyah veya beyaz yapar
    for i in range(h):
        for j in range(w):
            output[i, j] = 255 if gray_ch[i, j] > threshold else 0
            
    # Binary sonucu tekrar 3 kanallı (RGB) formatına getirir
    binary_3ch = np.stack([output, output, output], axis=2)
    return binary_3ch