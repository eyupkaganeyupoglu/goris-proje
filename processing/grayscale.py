import numpy as np

def apply_grayscale(img: np.ndarray) -> np.ndarray:
    # Hassas hesaplama yapabilmek için görüntüyü float32 tipine dönüştürür
    img_f = img.astype(np.float32)

    # Görüntünün mavi (B), yeşil (G) ve kırmızı (R) renk kanallarını ayırır
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]

    # Standart Luminosity formülü kullanarak gri tonlama değerini hesaplar
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    # Değerleri 0-255 aralığında tutar ve tekrar uint8 tipine dönüştürür
    Y = np.clip(Y, 0, 255).astype(np.uint8)

    # Elde edilen tek kanallı gri görüntüyü tekrar 3 kanallı yapıya getirir
    gray_3ch = np.stack([Y, Y, Y], axis=2)
    return gray_3ch