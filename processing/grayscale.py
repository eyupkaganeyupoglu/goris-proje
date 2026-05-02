import numpy as np

def apply_grayscale(img: np.ndarray) -> np.ndarray:
    # Veri tipi float32'ye dönüştürülür.
    img_f = img.astype(np.float32)
    
    # RGB'yi ayrıştırır.
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]
    
    # Standart luma formülü.
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Piksel değerleri sınırlandırılır ve uygun veri tipine dönüştürür.
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    
    # Tek kanallı veri 3 kanallı yapıya dönüştürür.
    gray_3ch = np.stack([Y, Y, Y], axis=2)
    
    return gray_3ch