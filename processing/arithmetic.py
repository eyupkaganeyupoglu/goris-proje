import numpy as np
from processing.zoom import apply_zoom

# Boyut eşitleme
def _match_dimensions(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    # Farklı boyutlardaki iki görüntüyü işleme sokabilmek için boyutlarını eşitler
    if img1.shape != img2.shape:
        scale_h = img1.shape[0] / img2.shape[0]
        scale_w = img1.shape[1] / img2.shape[1]
        scale = max(scale_h, scale_w)
        img2 = apply_zoom(img2, scale)
        img2 = img2[:img1.shape[0], :img1.shape[1]]
    return img2

# Görüntü toplama
def add_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    # İşlemden önce boyutları eşitler ve pikselleri float32 tipine çevirir
    img2 = _match_dimensions(img1, img2)
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    
    # Piksellerin ortalamasını alarak toplama (birleştirme) işlemini yapar
    result = (f1 + f2) / 2.0
    return np.clip(result, 0, 255).astype(np.uint8)

# Görüntü bölme
def divide_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    # Boyutları eşitler ve float32 tipine dönüşüm yapar
    img2 = _match_dimensions(img1, img2)
    f1 = img1.astype(np.float32)
    f2 = img2.astype(np.float32)
    
    # Sıfıra bölme hatasını önleyerek bölme işlemini gerçekleştirir
    safe_f2 = np.where(f2 == 0, 1e-6, f2)
    result = f1 / safe_f2
    return np.clip(result, 0, 255).astype(np.uint8)