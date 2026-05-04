import math
import numpy as np

def apply_rotation(img: np.ndarray, angle_deg: float = 45.0) -> np.ndarray:
    # Orijinal görüntünün boyutlarını ve merkez koordinatlarını alır
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    
    # Açıyı radian birimine çevirir ve trigonometrik değerleri hesaplar
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Döndürme sonrası görüntünün sığması için gereken yeni boyutları hesaplar
    new_w = int(math.ceil(abs(w * cos_a) + abs(h * sin_a)))
    new_h = int(math.ceil(abs(w * sin_a) + abs(h * cos_a)))
    new_cx, new_cy = new_w / 2.0, new_h / 2.0
    
    # Sonuç görüntüsü için uygun boyutta boş bir matris oluşturur
    if img.ndim == 3:
        output = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    else:
        output = np.zeros((new_h, new_w), dtype=img.dtype)
        
    # Her bir hedef piksel için ters haritalama (inverse mapping) yapar
    for i in range(new_h):
        for j in range(new_w):
            # Pikselin yeni merkeze göre konumunu belirler
            dy = i - new_cy
            dx = j - new_cx
            
            # Ters döndürme formülüyle orijinal görüntüdeki kaynak koordinatları bulur
            src_x = cos_a * dx + sin_a * dy + cx
            src_y = -sin_a * dx + cos_a * dy + cy
            
            # En yakın piksel değerini bulmak için koordinatları yuvarlar
            si = int(round(src_y))
            sj = int(round(src_x))
            
            # Kaynak koordinatlar orijinal görüntü sınırları içindeyse değeri kopyalar
            if 0 <= si < h and 0 <= sj < w:
                output[i, j] = img[si, sj]
                
    return output