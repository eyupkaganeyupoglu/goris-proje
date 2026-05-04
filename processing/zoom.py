import numpy as np

def _cubic_weight(t):
    # Bicubic interpolasyon için ağırlık değerini (kernel) hesaplayan yardımcı fonksiyondur
    t = abs(t)
    if t <= 1.0:
        return 1.5 * t**3 - 2.5 * t**2 + 1.0
    elif t <= 2.0:
        return -0.5 * t**3 + 2.5 * t**2 - 4.0 * t + 2.0
    else:
        return 0.0

# Nearest Neighbor (En Yakın Komşu)
def apply_zoom_nearest(img: np.ndarray, scale: float) -> np.ndarray:
    # Ölçek değerine göre yeni görüntü boyutlarını hesaplar
    h, w = img.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    # Sonuç görüntüsü için boş bir matris oluşturur
    output = np.zeros((new_h, new_w, img.shape[2]), dtype=np.uint8)
    total = new_h * new_w
    
    # Her hedef piksel için orijinal görüntüdeki en yakın koordinatı bulur
    for i in range(new_h):
        src_y = i / scale
        si = min(int(round(src_y)), h - 1)
        for j in range(new_w):
            src_x = j / scale
            sj = min(int(round(src_x)), w - 1)
            # En yakın pikselin değerini doğrudan kopyalar
            output[i, j] = img[si, sj]
        print(f"\rNearest Neighbor Progress: {(i + 1) * new_w}/{total} pixels", end="")
    print()
    return output

# Bilinear (Çift Doğrusal)
def apply_zoom_bilinear(img: np.ndarray, scale: float) -> np.ndarray:
    # Yeni boyutları ve hassas hesaplama için float matrisini hazırlar
    h, w = img.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    img_f = img.astype(np.float64)
    output = np.zeros((new_h, new_w, img.shape[2]), dtype=np.float64)
    total = new_h * new_w
    
    # Her hedef piksel için etrafındaki 4 komşu pikselin ağırlıklı ortalamasını alır
    for i in range(new_h):
        src_y = i / scale
        y0 = int(np.floor(src_y))
        y1 = min(y0 + 1, h - 1)
        dy = src_y - y0
        y0 = min(y0, h - 1)
        
        for j in range(new_w):
            src_x = j / scale
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, w - 1)
            dx = src_x - x0
            x0 = min(x0, w - 1)
            
            # Yatay ve dikey eksende doğrusal interpolasyon yapar
            top    = img_f[y0, x0] * (1 - dx) + img_f[y0, x1] * dx
            bottom = img_f[y1, x0] * (1 - dx) + img_f[y1, x1] * dx
            output[i, j] = top * (1 - dy) + bottom * dy
            
        print(f"\rBilinear Progress: {(i + 1) * new_w}/{total} pixels", end="")
    print()
    # Sonucu 0-255 arasına sınırlar ve uint8 formatına döner
    return np.clip(output, 0, 255).astype(np.uint8)

# Bicubic (Bikübik)
def apply_zoom_bicubic(img: np.ndarray, scale: float) -> np.ndarray:
    # Yeni boyutları ve float matrisini hazırlar
    h, w = img.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    img_f = img.astype(np.float64)
    output = np.zeros((new_h, new_w, img.shape[2]), dtype=np.float64)
    total = new_h * new_w
    
    # Her hedef piksel için 4x4 (toplam 16) komşu pikselin ağırlıklı toplamını hesaplar
    for i in range(new_h):
        src_y = i / scale
        y0 = int(np.floor(src_y))
        dy = src_y - y0
        for j in range(new_w):
            src_x = j / scale
            x0 = int(np.floor(src_x))
            dx = src_x - x0
            pixel = np.zeros(img.shape[2], dtype=np.float64)
            
            # 16 piksellik komşuluğu tarar ve kübik ağırlıkları uygular
            for m in range(-1, 3):
                wy = _cubic_weight(dy - m)
                yy = min(max(y0 + m, 0), h - 1)
                for n in range(-1, 3):
                    wx = _cubic_weight(dx - n)
                    xx = min(max(x0 + n, 0), w - 1)
                    pixel += wy * wx * img_f[yy, xx]
            output[i, j] = pixel
            
        print(f"\rBicubic Progress: {(i + 1) * new_w}/{total} pixels", end="")
    print()
    return np.clip(output, 0, 255).astype(np.uint8)

# Dispatcher
def apply_zoom(img: np.ndarray, scale: float = 2.0, method: str = "nearest") -> np.ndarray:
    # Seçilen yönteme (nearest, bilinear, bicubic) göre ilgili fonksiyonu çağırır
    method = method.lower()
    if method == "bilinear":
        return apply_zoom_bilinear(img, scale)
    elif method == "bicubic":
        return apply_zoom_bicubic(img, scale)
    else:
        return apply_zoom_nearest(img, scale)