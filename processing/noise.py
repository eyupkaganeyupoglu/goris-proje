import numpy as np

# Tuz-Biber gürültüsü ekleme
def add_salt_pepper(img: np.ndarray, amount: float = 0.05) -> np.ndarray:
    # Görüntü boyutlarına göre eklenecek toplam gürültü piksel sayısını hesaplar
    result = img.copy()
    h, w = img.shape[:2]
    total = h * w
    n_noise = int(total * amount)

    rng = np.random.default_rng()

    # Belirlenen miktar kadar rastgele noktaya beyaz (salt) pikseller ekler
    coords_salt = rng.integers(0, [h, w], size=(n_noise // 2, 2))
    for y, x in coords_salt:
        result[y, x] = 255

    # Belirlenen miktar kadar rastgele noktaya siyah (pepper) pikseller ekler
    coords_pepper = rng.integers(0, [h, w], size=(n_noise // 2, 2))
    for y, x in coords_pepper:
        result[y, x] = 0

    return result


# Median filtresiyle gürültü temizleme
def clean_median(img: np.ndarray, size: int = 3) -> np.ndarray:
    # Filtre boyutu için padding miktarını hesaplar ve sonuç matrisini hazırlar
    pad = size // 2
    result = np.zeros_like(img)

    # Renkli görüntülerde (3 kanallı) her kanal için ayrı işlem yapar
    if img.ndim == 3:
        h, w, c = img.shape
        for ch in range(c):
            channel = img[:, :, ch].astype(np.uint8)
            padded = np.pad(channel, pad, mode='edge')
            out_ch = np.zeros((h, w), dtype=np.uint8)
            
            # Her piksel için komşuluk penceresindeki değerleri sıralayıp ortancayı (median) seçer
            for i in range(h):
                for j in range(w):
                    window = padded[i:i + size, j:j + size].flatten().tolist()
                    window.sort()
                    out_ch[i, j] = window[len(window) // 2]
            result[:, :, ch] = out_ch
    # Tek kanallı görüntüler için median filtreleme uygular
    else:
        h, w = img.shape
        padded = np.pad(img, pad, mode='edge')
        for i in range(h):
            for j in range(w):
                window = padded[i:i + size, j:j + size].flatten().tolist()
                window.sort()
                result[i, j] = window[len(window) // 2]
    return result