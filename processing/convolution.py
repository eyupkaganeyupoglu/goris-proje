import numpy as np

# Konvolüsyon işlemi
def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Görüntü ve kernel boyutlarını alıp kenar boşluklarını (padding) hesaplar
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Görüntü kenarlarını sıfırla doldurur (padding)
    padded = np.pad(image.astype(np.float32), ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Convolution işlemi için kernel matrisini ters çevirir (flip)
    flipped_kernel = np.flip(kernel)
    output = np.zeros((h, w), dtype=np.float32)
    
    # Kernel'ı görüntü üzerinde gezdirerek ağırlıklı toplamı hesaplar
    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * flipped_kernel)
            
    return np.clip(output, 0, 255).astype(np.uint8)

# Ortalama filtresi
def apply_mean_filter(img: np.ndarray, size: int = 3) -> np.ndarray:
    # Belirlenen boyutta tüm değerleri eşit olan bir kernel oluşturur
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    
    # Görüntü kanal yapısına göre convolution işlemini uygular
    if img.ndim == 2:
        return convolve(img, kernel)
    channels = [convolve(img[:, :, c], kernel) for c in range(img.shape[2])]
    return np.stack(channels, axis=2)