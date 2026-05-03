import numpy as np

def apply_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]

    if x1 >= x2 or y1 >= y2:
        raise ValueError("X1 ve Y1 değerleri sırasıyla X2 ve Y2 değerlerinden küçük olmalıdır.")

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"Koordinatlar resim sınırları (Genişlik: {w}, Yükseklik: {h}) içinde olmalıdır.")

    return img[y1:y2, x1:x2]