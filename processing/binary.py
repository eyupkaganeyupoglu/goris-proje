import numpy as np
from processing.grayscale import apply_grayscale

def apply_binary(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    # Görseli gri tonlamaya dönüştürür.
    gray = apply_grayscale(img)
    
    # Tek kanallı veriyi ayrıştırır.
    gray_ch = gray[:, :, 0].astype(np.float32)
    h, w = gray_ch.shape
    
    # Belirlenen boyutta boş bir matris oluşturur.
    output = np.zeros((h, w), dtype=np.uint8)
    
    # Pikselleri eşik değerine göre siyah veya beyaz yapar.
    for i in range(h):
        for j in range(w):
            output[i, j] = 255 if gray_ch[i, j] > threshold else 0
            
    # Sonucu 3 kanallı yapıya dönüştürür.
    binary_3ch = np.stack([output, output, output], axis=2)
    
    return binary_3ch