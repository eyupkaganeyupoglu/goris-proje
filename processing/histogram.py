import numpy as np
from processing.grayscale import apply_grayscale

# Histogram hesaplama
def compute_histogram(img: np.ndarray) -> list:
    # Görüntüyü gri tonlamaya çevirir ve tek kanal üzerinden boyutlarını alır
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    
    # Her pikselin yoğunluk değerini sayarak histogram listesini oluşturur
    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(channel[i, j])] += 1
    return hist

# Histogram germe
def apply_histogram_stretch(img: np.ndarray, a: int = 0, b: int = 255) -> np.ndarray:
    # Görüntüdeki minimum (c) ve maksimum (d) yoğunluk değerlerini belirler
    img_f = img.astype(np.float32)
    c = float(img_f.min())
    d = float(img_f.max())
    
    if d == c:
        return img
    
    # Yoğunluk değerlerini [a, b] aralığına doğrusal olarak yayar (stretching)
    stretched = (img_f - c) * ((b - a) / (d - c)) + a
    return np.clip(stretched, 0, 255).astype(np.uint8)

# Histogram eşitleme
def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    # Görüntüyü gri tonlamaya çevirir ve toplam piksel sayısını hesaplar
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    total_pixels = h * w

    # Görüntünün histogramını (yoğunluk dağılımını) hesaplar
    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(channel[i, j])] += 1

    # Kümülatif dağılım fonksiyonunu (CDF) hesaplar
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # Sıfırdan farklı en küçük CDF değerini bulur
    cdf_min = 0
    for i in range(256):
        if cdf[i] != 0:
            cdf_min = cdf[i]
            break

    # Histogram eşitleme formülüyle yeni değerler için bir lookup table oluşturur
    lookup = [0] * 256
    denom = total_pixels - cdf_min
    if denom == 0:
        return gray
    for i in range(256):
        lookup[i] = int(round((cdf[i] - cdf_min) / denom * 255))

    # Orijinal pikselleri lookup table üzerinden yeni değerleriyle eşleştirir
    output = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            val = lookup[int(channel[i, j])]
            output[i, j] = [val, val, val]

    return output.astype(np.uint8)

# Histogram genişletme
def apply_histogram_expand(img: np.ndarray, a: float = 0.3, b: float = 0.7) -> np.ndarray:
    # Görüntüyü [0, 1] aralığına normalize eder
    img_f = img.astype(np.float32) / 255.0

    lo = float(a)
    hi = float(b)
    if hi <= lo:
        return img
    
    # Belirlenen [a, b] aralığındaki değerleri tüm aralığa genişletir (expand)
    expanded = (1.0 / (hi - lo)) * (img_f - lo)
    expanded = np.clip(expanded, 0.0, 1.0)
    return (expanded * 255.0).astype(np.uint8)