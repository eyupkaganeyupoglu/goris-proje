import numpy as np

def apply_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]

    # Başlangıç ve bitiş koordinatlarının sırasını düzeltir.
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # Koordinatları resim boyutları dahilinde sınırlandırır.
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Kırpma alanının en az 1x1 boyutunda olmasını sağlar.
    if x1 == x2:
        x2 = min(x1 + 1, w)
        if x1 == x2:
            x1 = max(0, x2 - 1)
    if y1 == y2:
        y2 = min(y1 + 1, h)
        if y1 == y2:
            y1 = max(0, y2 - 1)

    # Belirlenen aralıkta crop yapar ve sonucu döndürür.
    return img[y1:y2, x1:x2]
