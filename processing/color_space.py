import numpy as np
from processing.grayscale import apply_grayscale

def _split_bgr(img: np.ndarray):
    # Görüntüyü float32 tipine çevirip R, G ve B kanallarını ayırır
    img_f = img.astype(np.float32)
    B = img_f[:, :, 0]
    G = img_f[:, :, 1]
    R = img_f[:, :, 2]
    return R, G, B

# RGB to Grayscale
def apply_grayscale_conversion(img: np.ndarray) -> np.ndarray:
    # Kod tekrarını önlemek için ana grayscale modülündeki fonksiyonu çağırır
    return apply_grayscale(img)

# RGB to NTSC
def apply_ntsc_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü NTSC (YIQ) renk uzayına dönüştürür
    R, G, B = _split_bgr(img)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B

    # Bileşenleri normalize ederek 0-255 aralığına ölçekler
    I_norm = (I + 0.5957) * (255.0 / 1.1914)
    Q_norm = (Q + 0.5226) * (255.0 / 1.0452)

    Y = np.clip(Y, 0, 255)
    I_norm = np.clip(I_norm, 0, 255)
    Q_norm = np.clip(Q_norm, 0, 255)

    return np.stack([Q_norm, I_norm, Y], axis=2).astype(np.uint8)

# RGB to YCbCr
def apply_ycbcr_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü YCbCr renk uzayına dönüştürür
    R, G, B = _split_bgr(img)

    Y  =  0.257 * R + 0.504 * G + 0.098 * B + 16
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128
    Cr =  0.439 * R - 0.368 * G - 0.071 * B + 128

    Y  = np.clip(Y,  0, 255)
    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)

    return np.stack([Cr, Cb, Y], axis=2).astype(np.uint8)

# RGB to CMY
def apply_cmy_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü CMY renk uzayına çevirir
    R, G, B = _split_bgr(img)

    C = 255.0 - R
    M = 255.0 - G
    Y = 255.0 - B

    return np.stack([Y, M, C], axis=2).astype(np.uint8)

# RGB to CMYK
def apply_cmyk_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü CMYK renk uzayına dönüştürür
    R, G, B = _split_bgr(img)

    Rn = R / 255.0
    Gn = G / 255.0
    Bn = B / 255.0

    # Key (K - siyah) değerini hesaplar
    K = 1.0 - np.maximum(np.maximum(Rn, Gn), Bn)

    # Sıfıra bölmeyi önleyerek C, M, Y bileşenlerini hesaplar
    denom = 1.0 - K
    denom_safe = np.where(denom == 0, 1.0, denom)

    C = (1.0 - Rn - K) / denom_safe
    M = (1.0 - Gn - K) / denom_safe
    Y_ch = (1.0 - Bn - K) / denom_safe

    # Saf siyah piksellerde değerleri sıfırlar
    C = np.where(denom == 0, 0.0, C)
    M = np.where(denom == 0, 0.0, M)
    Y_ch = np.where(denom == 0, 0.0, Y_ch)

    # Değerleri 0-255 aralığına ölçekler
    C = np.clip(C * 255, 0, 255)
    M = np.clip(M * 255, 0, 255)
    Y_ch = np.clip(Y_ch * 255, 0, 255)

    return np.stack([Y_ch, M, C], axis=2).astype(np.uint8)

# RGB to HSI
def apply_hsi_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü HSI (Hue, Saturation, Intensity) renk uzayına çevirir
    R, G, B = _split_bgr(img)

    # Yoğunluk (Intensity) değerini hesaplar
    I = (R + G + B) / 3.0

    # Doygunluk (Saturation) değerini hesaplar
    min_rgb = np.minimum(np.minimum(R, G), B)
    I_safe = np.where(I == 0, 1.0, I)
    S = 1.0 - min_rgb / I_safe
    S = np.where(I == 0, 0.0, S)

    # Renk Özü (Hue) değerini trigonometrik formülle hesaplar
    rg = R - G
    rb = R - B
    gb = G - B

    numer = 0.5 * (rg + rb)
    denom = np.sqrt(rg * rg + rb * gb + 1e-10)
    theta = np.arccos(np.clip(numer / denom, -1.0, 1.0))

    H = np.where(B <= G, theta, 2.0 * np.pi - theta)

    # Radyan cinsinden açıyı dereceye ve 0-255 aralığına çevirir
    H_deg = H * (180.0 / np.pi)
    H_scaled = H_deg * (255.0 / 360.0)

    H_out = np.clip(H_scaled, 0, 255)
    S_out = np.clip(S * 255.0, 0, 255)
    I_out = np.clip(I, 0, 255)

    return np.stack([I_out, S_out, H_out], axis=2).astype(np.uint8)

# RGB to CIE XYZ
def apply_xyz_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü CIE XYZ renk uzayına dönüştürür
    R, G, B = _split_bgr(img)

    # Lineerleştirme (Gamma Correction tersi) uygular
    def linearize(ch):
        c = ch / 255.0
        return np.where(c > 0.04045,
                        ((c + 0.055) / 1.055) ** 2.4,
                        c / 12.92)

    Rl = linearize(R)
    Gl = linearize(G)
    Bl = linearize(B)

    # Lineer RGB'den XYZ bileşenlerine geçiş yapar
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl

    # XYZ değerlerini D65 referans beyazına göre normalize ve ölçekleme yapar
    X_scaled = np.clip(X / 0.95047 * 255, 0, 255)
    Y_scaled = np.clip(Y / 1.00000 * 255, 0, 255)
    Z_scaled = np.clip(Z / 1.08883 * 255, 0, 255)

    return np.stack([Z_scaled, Y_scaled, X_scaled], axis=2).astype(np.uint8)

# RGB to CIE L*a*b*
def apply_lab_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü CIE L*a*b* renk uzayına dönüştürür
    R, G, B = _split_bgr(img)

    # Lineerleştirme (Gamma Correction tersi) uygular
    def linearize(ch):
        c = ch / 255.0
        return np.where(c > 0.04045,
                        ((c + 0.055) / 1.055) ** 2.4,
                        c / 12.92)

    Rl = linearize(R)
    Gl = linearize(G)
    Bl = linearize(B)

    # XYZ bileşenlerini hesaplar
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl

    # D65 referans beyazı ve Lab dönüşüm katsayıları
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    delta = 6.0 / 29.0
    delta_sq = delta * delta
    delta_cb = delta * delta * delta

    # Lab dönüşümünde kullanılan yardımcı fonksiyon
    def f(t):
        return np.where(t > delta_cb, np.power(t, 1.0 / 3.0), t / (3.0 * delta_sq) + 4.0 / 29.0)

    fx = f(X / Xn)
    fy = f(Y / Yn)
    fz = f(Z / Zn)

    # Parlaklık (L) ve renk karşıtlığı (a, b) değerlerini hesaplar
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    # Görselleştirme için değerleri 0-255 aralığına getirir
    L_out = np.clip(L * 255.0 / 100.0, 0, 255)
    a_out = np.clip(a + 128, 0, 255)
    b_out = np.clip(b + 128, 0, 255)

    return np.stack([b_out, a_out, L_out], axis=2).astype(np.uint8)

# RGB to CIE L*u*v*
def apply_luv_conversion(img: np.ndarray) -> np.ndarray:
    # RGB görüntüyü CIE L*u*v* renk uzayına dönüştürür
    R, G, B = _split_bgr(img)

    # Lineerleştirme (Gamma Correction tersi) uygular
    def linearize(ch):
        c = ch / 255.0
        return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

    Rl = linearize(R)
    Gl = linearize(G)
    Bl = linearize(B)

    # XYZ bileşenlerini hesaplar
    X = 0.4124564 * Rl + 0.3575761 * Gl + 0.1804375 * Bl
    Y = 0.2126729 * Rl + 0.7151522 * Gl + 0.0721750 * Bl
    Z = 0.0193339 * Rl + 0.1191920 * Gl + 0.9503041 * Bl

    # Referans beyaz noktası ve katsayıları belirler
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    un_prime = 4.0 * Xn / (Xn + 15.0 * Yn + 3.0 * Zn)
    vn_prime = 9.0 * Yn / (Xn + 15.0 * Yn + 3.0 * Zn)

    # u' ve v' koordinatlarını sıfıra bölme kontrolü ile hesaplar
    denom = X + 15.0 * Y + 3.0 * Z
    denom_safe = np.where(denom == 0, 1.0, denom)

    u_prime = 4.0 * X / denom_safe
    v_prime = 9.0 * Y / denom_safe
    u_prime = np.where(denom == 0, 0.0, u_prime)
    v_prime = np.where(denom == 0, 0.0, v_prime)

    # Parlaklık (L) değerini hesaplar
    delta = 6.0 / 29.0
    delta_cb = delta ** 3

    yr = Y / Yn
    L = np.where(yr > delta_cb, 116.0 * np.power(yr, 1.0 / 3.0) - 16.0, 903.3 * yr)

    # Renk değerlerini (u, v) parlaklığa göre ölçeklendirir
    u_star = 13.0 * L * (u_prime - un_prime)
    v_star = 13.0 * L * (v_prime - vn_prime)

    # Değerleri görselleştirme için 0-255 aralığına getirir
    L_out = np.clip(L * 255.0 / 100.0, 0, 255)
    u_out = np.clip(u_star + 134, 0, 255)
    v_out = np.clip(v_star + 140, 0, 255)

    return np.stack([v_out, u_out, L_out], axis=2).astype(np.uint8)