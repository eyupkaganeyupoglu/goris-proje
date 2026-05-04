import numpy as np
from processing.threshold import apply_threshold

def apply_binary(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    # Eşikleme işlemini threshold modülündeki fonksiyonu kullanarak yapar
    return apply_threshold(img, threshold=threshold)