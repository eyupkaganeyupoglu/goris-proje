import numpy as np

def apply_zoom(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    img_f = img.astype(np.float32)
    output = np.zeros((new_h, new_w, img.shape[2]), dtype=np.float32)

    for i in range(new_h):
        for j in range(new_w):
            src_y = i / scale
            src_x = j / scale
            si = min(int(round(src_y)), h - 1)
            sj = min(int(round(src_x)), w - 1)
            output[i, j] = img_f[si, sj]

    return np.clip(output, 0, 255).astype(np.uint8)