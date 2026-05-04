import numpy as np

def apply_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    # Orijinal görüntünün yükseklik ve genişlik bilgilerini alır
    h, w = img.shape[:2]

    # Başlangıç ve bitiş koordinatlarının mantıksal doğruluğunu kontrol eder
    if x1 >= x2 or y1 >= y2:
        raise ValueError("X1 ve Y1 değerleri sırasıyla X2 ve Y2 değerlerinden küçük olmalıdır.")

    # Koordinatların görüntü sınırları içerisinde olup olmadığını kontrol eder
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"Koordinatlar resim sınırları (Genişlik: {w}, Yükseklik: {h}) içinde olmalıdır.")

    # Belirlenen koordinat aralığını dilimleyerek (slicing) görüntüyü kırpar
    return img[y1:y2, x1:x2]