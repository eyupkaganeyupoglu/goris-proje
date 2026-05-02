import numpy as np
from processing.grayscale import apply_grayscale


def compute_histogram(img: np.ndarray) -> list:
    """
    Piksel yoğunluklarının (0-255) frekansını manuel olarak döngü ile hesaplar.
    Girdi: Görüntü (numpy dizisi)
    Çıktı: İndis = yoğunluk, Değer = miktar şeklinde 256 elemanlı liste.
    """
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(channel[i, j])] += 1
    return hist


def apply_histogram_stretch(img: np.ndarray, a: int = 0, b: int = 255) -> np.ndarray:
    """
    Histogram Germe (Contrast Stretching)

    Görüntünün kontrastını artırmak amacıyla piksel değerlerini [a, b] aralığına
    doğrusal (lineer) olarak ölçekler. Düşük kontrastlı görüntülerde dar bir
    alana sıkışmış gri seviyelerini daha geniş bir alana yayarak detayların
    netleşmesini sağlar.

    Formül:
        P_çıkış = (P_giriş - c) * ((b - a) / (d - c)) + a

    Burada:
        a = Hedef minimum piksel değeri (varsayılan: 0)
        b = Hedef maksimum piksel değeri (varsayılan: 255)
        c = Görüntüdeki mevcut minimum piksel değeri
        d = Görüntüdeki mevcut maksimum piksel değeri
    """
    # Hassas hesaplama için float32'ye dönüştür
    img_f = img.astype(np.float32)

    # Görüntüdeki mevcut minimum (c) ve maksimum (d) değerlerini bul
    c = float(img_f.min())
    d = float(img_f.max())

    # Tüm pikseller aynı değerdeyse bölme hatasını önle
    if d == c:
        return img

    # Doğrusal germe formülü: P_çıkış = (P_giriş - c) * ((b - a) / (d - c)) + a
    stretched = (img_f - c) * ((b - a) / (d - c)) + a

    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Histogram Eşitleme — CDF (Kümülatif Dağılım Fonksiyonu) tabanlı algoritma.

    Görüntünün histogramını değiştirerek yoğunluğu tüm 0-255 aralığına mümkün
    olduğunca eşit dağıtır. Yüksek frekanslı seviyeleri geniş alana yayar,
    düşükleri birbirine yaklaştırır.
    """
    gray = apply_grayscale(img)
    channel = gray[:, :, 0]
    h, w = channel.shape
    total_pixels = h * w

    # 1. Manuel histogram hesaplama
    hist = [0] * 256
    for i in range(h):
        for j in range(w):
            hist[int(channel[i, j])] += 1

    # 2. CDF (Kümülatif Dağılım Fonksiyonu) hesaplama
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    # 3. CDF'yi normalleştirerek yoğunluk eşleme tablosu (lookup table) oluşturma
    cdf_min = 0
    for i in range(256):
        if cdf[i] != 0:
            cdf_min = cdf[i]
            break

    lookup = [0] * 256
    denom = total_pixels - cdf_min
    if denom == 0:
        return gray  # Tüm pikseller aynı değerdeyse orijinali döndür
    for i in range(256):
        lookup[i] = int(round((cdf[i] - cdf_min) / denom * 255))

    # 4. Yeni değerleri orijinal koordinatlara uygulama
    output = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            val = lookup[int(channel[i, j])]
            output[i, j] = [val, val, val]

    return output.astype(np.uint8)


def apply_histogram_expand(
    img: np.ndarray,
    a: float = 0.3,
    b: float = 0.7,
) -> np.ndarray:
    """
    Histogram Genişletme

    Görüntünün belirli bir parlaklık aralığını yayarak o bölgedeki ayrıntıların
    ortaya çıkarılmasını sağlar. Orijinal histogramın genel yapısı ve şekli
    korunurken, belirlenen aralıktaki pikseller tüm dinamik aralığa (0-255) yayılır.

    Formül:
        g(v) = ((L - 1) / (b - a)) * (f(v) - a)

    Burada:
        f(v) = Orijinal piksel değeri (0.0-1.0 normalize)
        a    = Genişletilecek alt sınır (0.0-1.0)
        b    = Genişletilecek üst sınır (0.0-1.0)
        L-1  = 255 (8-bit görüntü için maksimum değer)
        g(v) = Genişletilmiş piksel değeri
    """
    # 0-1 arasına normalize et
    img_f = img.astype(np.float32) / 255.0

    lo = float(a)
    hi = float(b)

    # Geçersiz aralık kontrolü
    if hi <= lo:
        return img

    # Histogram Genişletme formülü: g(v) = ((L-1) / (b - a)) * (f(v) - a)
    # L - 1 = 1.0 (normalleştirilmiş çıktıda), sonra 255'e ölçeklenir
    expanded = (1.0 / (hi - lo)) * (img_f - lo)

    # Değerleri 0-1 arasına kırp ve 0-255 ölçeğine geri getir
    expanded = np.clip(expanded, 0.0, 1.0)
    return (expanded * 255.0).astype(np.uint8)
