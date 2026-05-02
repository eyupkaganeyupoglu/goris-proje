import numpy as np

def _split_bgr(img: np.ndarray):
    # BGR görüntüyü float32'ye çevirip R, G, B kanallarını döndürür.
    img_f = img.astype(np.float32)
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]
    return R, G, B


# RGB to Grayscale
def apply_grayscale_conversion(img: np.ndarray) -> np.ndarray:
    # Y = 0.299·R + 0.587·G + 0.114·B
    R, G, B = _split_bgr(img)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    # Görselleştirme için 3 kanallı gri görüntü
    return np.stack([Y, Y, Y], axis=2)


# RGB to NTSC
def apply_ntsc_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    # Normalize: I -> [0,255], Q -> [0,255]
    I_norm = (I + 0.5957) * (255.0 / 1.1914)
    Q_norm = (Q + 0.5226) * (255.0 / 1.0452)

    Y = np.clip(Y, 0, 255)
    I_norm = np.clip(I_norm, 0, 255)
    Q_norm = np.clip(Q_norm, 0, 255)

    # BGR sırasıyla kaydedilir: B=Q, G=I, R=Y
    return np.stack([Q_norm, I_norm, Y], axis=2).astype(np.uint8)


# RGB to YCbCr
def apply_ycbcr_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    Y  =  0.257 * R + 0.504 * G + 0.098 * B + 16
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    Cr =  0.439 * R - 0.368 * G - 0.071 * B + 128

    Y  = np.clip(Y,  0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)

    # BGR sırası: B=Cr, G=Cb, R=Y
    return np.stack([Cr, Cb, Y], axis=2).astype(np.uint8)


# RGB to CMY
def apply_cmy_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    C = 255.0 - R
    M = 255.0 - G
    Y = 255.0 - B

    # BGR sırası: B=Y, G=M, R=C
    return np.stack([Y, M, C], axis=2).astype(np.uint8)


# RGB to CMYK
def apply_cmyk_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    Rn = R / 255.0
    Gn = G / 255.0
    Bn = B / 255.0

    K = 1.0 - np.maximum(np.maximum(Rn, Gn), Bn)

    # Sıfıra bölmeyi önle
    denom = 1.0 - K
    denom_safe = np.where(denom == 0, 1.0, denom)

    C = (1.0 - Rn - K) / denom_safe
    M = (1.0 - Gn - K) / denom_safe
    Y_ch = (1.0 - Bn - K) / denom_safe

    # K=1 (saf siyah) piksellerde C=M=Y=0
    C = np.where(denom == 0, 0.0, C)
    M = np.where(denom == 0, 0.0, M)
    Y_ch = np.where(denom == 0, 0.0, Y_ch)

    # [0, 255] aralığına ölçekle
    C = np.clip(C * 255, 0, 255)
    M = np.clip(M * 255, 0, 255)
    Y_ch = np.clip(Y_ch * 255, 0, 255)

    # BGR: B=Y, G=M, R=C
    return np.stack([Y_ch, M, C], axis=2).astype(np.uint8)


# RGB to HSI
def apply_hsi_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    # I=0 durumunu engelle
    I_safe = np.where(I == 0, 1.0, I)
    S = 1.0 - min_rgb / I_safe
    S = np.where(I == 0, 0.0, S)

    # Hue
    rg = R - G
    rb = R - B
    gb = G - B

    numer = 0.5 * (rg + rb)
    denom = np.sqrt(rg * rg + rb * gb + 1e-10)  # epsilon ile sıfıra bölmeden kaçın
    theta = np.arccos(np.clip(numer / denom, -1.0, 1.0))

    H = np.where(B <= G, theta, 2.0 * np.pi - theta)

    # Radyan -> derece -> [0, 255] aralığına ölçekle
    H_deg = H * (180.0 / np.pi)
    H_scaled = H_deg * (255.0 / 360.0)

    H_out = np.clip(H_scaled, 0, 255)
    S_out = np.clip(S * 255.0, 0, 255)
    I_out = np.clip(I, 0, 255)

    # BGR: B=I, G=S, R=H
    return np.stack([I_out, S_out, H_out], axis=2).astype(np.uint8)


# RGB to CIE XYZ
def apply_xyz_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    # sRGB -> lineer
    def linearize(ch):
        c = ch / 255.0
        return np.where(c > 0.04045,
                        ((c + 0.055) / 1.055) ** 2.4,
                        c / 12.92)

    Rl = linearize(R)
    Gl = linearize(G)
    Bl = linearize(B)

    # Lineer RGB -> XYZ
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl

    # XYZ [0, ~1] aralığında -> [0, 255] olarak ölçekle
    # D65 referans: Xn=0.95047, Yn=1.0, Zn=1.08883
    X_scaled = np.clip(X / 0.95047 * 255, 0, 255)
    Y_scaled = np.clip(Y / 1.00000 * 255, 0, 255)
    Z_scaled = np.clip(Z / 1.08883 * 255, 0, 255)

    # BGR: B=Z, G=Y, R=X
    return np.stack([Z_scaled, Y_scaled, X_scaled], axis=2).astype(np.uint8)


# RGB to CIE L*a*b*
def apply_lab_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    # sRGB -> lineer
    def linearize(ch):
        c = ch / 255.0
        return np.where(c > 0.04045,
                        ((c + 0.055) / 1.055) ** 2.4,
                        c / 12.92)

    Rl = linearize(R)
    Gl = linearize(G)
    Bl = linearize(B)

    # Lineer RGB -> XYZ
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl

    # D65 referans beyazı
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    # CIE f() fonksiyonu
    delta = 6.0 / 29.0
    delta_sq = delta * delta
    delta_cb = delta * delta * delta

    def f(t):
        return np.where(t > delta_cb, np.power(t, 1.0 / 3.0), t / (3.0 * delta_sq) + 4.0 / 29.0)

    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)

    L = 116.0 * fy - 16.0    # [0, 100]
    a = 500.0 * (fx - fy)    # ≈[−128, 127]
    b = 200.0 * (fy - fz)    # ≈[−128, 127]

    # Görselleştirme için ölçekle
    L_out = np.clip(L * 255.0 / 100.0, 0, 255)
    a_out = np.clip(a + 128, 0, 255)
    b_out = np.clip(b + 128, 0, 255)

    # BGR: B=b*, G=a*, R=L*
    return np.stack([b_out, a_out, L_out], axis=2).astype(np.uint8)


# RGB to CIE L*u*v*
def apply_luv_conversion(img: np.ndarray) -> np.ndarray:
    R, G, B = _split_bgr(img)

    # sRGB -> lineer
    def linearize(ch):
        c = ch / 255.0
        return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

    Rl = linearize(R)
    Gl = linearize(G)
    Bl = linearize(B)

    # Lineer RGB -> XYZ
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl

    # D65 referans beyazı
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    # Referans u'n, v'n
    un_prime = 4.0 * Xn / (Xn + 15.0 * Yn + 3.0 * Zn)
    vn_prime = 9.0 * Yn / (Xn + 15.0 * Yn + 3.0 * Zn)

    # u', v' hesapla (sıfıra bölmeden korun)
    denom = X + 15.0 * Y + 3.0 * Z
    denom_safe = np.where(denom == 0, 1.0, denom)

    u_prime = 4.0 * X / denom_safe
    v_prime = 9.0 * Y / denom_safe
    u_prime = np.where(denom == 0, 0.0, u_prime)
    v_prime = np.where(denom == 0, 0.0, v_prime)

    # L* hesapla
    delta = 6.0 / 29.0
    delta_cb = delta ** 3

    yr = Y / Yn
    L = np.where(yr > delta_cb, 116.0 * np.power(yr, 1.0 / 3.0) - 16.0, 903.3 * yr)

    # u*, v*
    u_star = 13.0 * L * (u_prime - un_prime)
    v_star = 13.0 * L * (v_prime - vn_prime)

    # Görselleştirme: L* -> [0,255], u* ve v* -> offset +134 -> [0,255]
    L_out = np.clip(L * 255.0 / 100.0, 0, 255)
    u_out = np.clip(u_star + 134, 0, 255)
    v_out = np.clip(v_star + 140, 0, 255)

    # BGR: B=v*, G=u*, R=L*
    return np.stack([v_out, u_out, L_out], axis=2).astype(np.uint8)