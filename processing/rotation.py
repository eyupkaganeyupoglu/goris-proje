import math
import numpy as np

def apply_rotation(img: np.ndarray, angle_deg: float = 45.0) -> np.ndarray:
    # Görselin orijinal boyutunu ve merkez koordinatlarını belirler.
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    
    # Açıyı radyana dönüştürür ve trigonometrik değerlerini hesaplar.
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Döndürülmüş görselin sığacağı yeni canvas boyutunu hesaplar.
    new_w = int(math.ceil(abs(w * cos_a) + abs(h * sin_a)))
    new_h = int(math.ceil(abs(w * sin_a) + abs(h * cos_a)))

    new_cx, new_cy = new_w / 2.0, new_h / 2.0

    # Uygun kanal yapısına göre boş bir çıktı matrisi, yani yeni canvası oluşturur.
    if img.ndim == 3:
        output = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    else:
        output = np.zeros((new_h, new_w), dtype=img.dtype)

    for i in range(new_h):
        for j in range(new_w):
            # Yeni merkezden olan uzaklığı hesaplar.
            dy = i - new_cy
            dx = j - new_cx
            
            # Ters döndürme işlemi ile orijinal görsel koordinatlarını bulur.
            src_x = cos_a * dx + sin_a * dy + cx
            src_y = -sin_a * dx + cos_a * dy + cy
            
            si = int(round(src_y))
            sj = int(round(src_x))
            
            # Geçerli koordinatları yeni matrise, yani canvasa aktarır.
            if 0 <= si < h and 0 <= sj < w:
                output[i, j] = img[si, sj]
                
    return output