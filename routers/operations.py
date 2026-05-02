import os
import uuid
import io

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from processing.grayscale import apply_grayscale
from processing.binary import apply_binary
from processing.rotation import apply_rotation
from processing.crop import apply_crop
from processing.zoom import apply_zoom
from processing.zoom import apply_zoom
from processing.histogram import compute_histogram, apply_histogram_stretch, apply_histogram_equalization, apply_histogram_expand

from processing.arithmetic import add_images, divide_images
from processing.contrast import apply_contrast_multiply, apply_contrast_log
from processing.convolution import apply_mean_filter
from processing.threshold import apply_threshold
from processing.edge_detection import apply_edge_prewitt
from processing.noise import add_salt_pepper, clean_mean, clean_median
from processing.sharpening import apply_sharpening
from processing.morphology import apply_dilate, apply_erode, apply_opening, apply_closing

router = APIRouter()
RESULT_DIR = "static/temp_results"
os.makedirs(RESULT_DIR, exist_ok=True)


def _decode(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")
    return img


def _save(result: np.ndarray, prefix: str) -> str:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, result)
    return f"/{path.replace(os.sep, '/')}"


# ── 1. Grayscale ─────────────────────────────────────────────────────────────
@router.post("/process/grayscale")
async def grayscale(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_grayscale(img)
    return JSONResponse({"result_url": _save(result, "grayscale")})


# ── 2. Binary ────────────────────────────────────────────────────────────────
@router.post("/process/binary")
async def binary(file: UploadFile = File(...), threshold: int = Form(128)):
    img = _decode(await file.read())
    result = apply_binary(img, threshold=threshold)
    return JSONResponse({"result_url": _save(result, "binary")})


# ── 3. Rotation ──────────────────────────────────────────────────────────────
@router.post("/process/rotation")
async def rotation(file: UploadFile = File(...), angle: float = Form(45.0)):
    img = _decode(await file.read())
    result = apply_rotation(img, angle_deg=angle)
    return JSONResponse({"result_url": _save(result, "rotation")})


# ── 4. Crop ──────────────────────────────────────────────────────────────────
@router.post("/process/crop")
async def crop(
    file: UploadFile = File(...),
    x1: int = Form(0), y1: int = Form(0),
    x2: int = Form(100), y2: int = Form(100)
):
    img = _decode(await file.read())
    h, w = img.shape[:2]

    if x1 >= x2 or y1 >= y2:
        return JSONResponse(
            status_code=400,
            content={"detail": "X1 ve Y1 değerleri sırasıyla X2 ve Y2 değerlerinden küçük olmalıdır."}
        )

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Koordinatlar resim sınırları (Genişlik: {w}, Yükseklik: {h}) içinde olmalıdır."}
        )

    result = apply_crop(img, x1=x1, y1=y1, x2=x2, y2=y2)
    return JSONResponse({"result_url": _save(result, "crop")})


# ── 5. Zoom ──────────────────────────────────────────────────────────────────
@router.post("/process/zoom")
async def zoom(file: UploadFile = File(...), scale: float = Form(2.0), method: str = Form("nearest")):
    img = _decode(await file.read())
    result = apply_zoom(img, scale=scale, method=method)
    return JSONResponse({"result_url": _save(result, "zoom")})


# ── 6. Color Space Conversions ───────────────────────────────────────────────
@router.post("/process/ntsc")
async def ntsc(file: UploadFile = File(...)):
    from processing.color_space import apply_ntsc_conversion
    img = _decode(await file.read())
    result = apply_ntsc_conversion(img)
    return JSONResponse({"result_url": _save(result, "ntsc")})

@router.post("/process/ycbcr")
async def ycbcr(file: UploadFile = File(...)):
    from processing.color_space import apply_ycbcr_conversion
    img = _decode(await file.read())
    result = apply_ycbcr_conversion(img)
    return JSONResponse({"result_url": _save(result, "ycbcr")})

@router.post("/process/cmy")
async def cmy(file: UploadFile = File(...)):
    from processing.color_space import apply_cmy_conversion
    img = _decode(await file.read())
    result = apply_cmy_conversion(img)
    return JSONResponse({"result_url": _save(result, "cmy")})

@router.post("/process/cmyk")
async def cmyk(file: UploadFile = File(...)):
    from processing.color_space import apply_cmyk_conversion
    img = _decode(await file.read())
    result = apply_cmyk_conversion(img)
    return JSONResponse({"result_url": _save(result, "cmyk")})

@router.post("/process/hsi")
async def hsi(file: UploadFile = File(...)):
    from processing.color_space import apply_hsi_conversion
    img = _decode(await file.read())
    result = apply_hsi_conversion(img)
    return JSONResponse({"result_url": _save(result, "hsi")})

@router.post("/process/xyz")
async def xyz(file: UploadFile = File(...)):
    from processing.color_space import apply_xyz_conversion
    img = _decode(await file.read())
    result = apply_xyz_conversion(img)
    return JSONResponse({"result_url": _save(result, "xyz")})

@router.post("/process/lab")
async def lab(file: UploadFile = File(...)):
    from processing.color_space import apply_lab_conversion
    img = _decode(await file.read())
    result = apply_lab_conversion(img)
    return JSONResponse({"result_url": _save(result, "lab")})

@router.post("/process/luv")
async def luv(file: UploadFile = File(...)):
    from processing.color_space import apply_luv_conversion
    img = _decode(await file.read())
    result = apply_luv_conversion(img)
    return JSONResponse({"result_url": _save(result, "luv")})


# ── 7. Histogram (returns frequencies as JSON) ───────────────────────────────
@router.post("/process/histogram")
async def histogram(file: UploadFile = File(...)):
    img = _decode(await file.read())
    hist = compute_histogram(img)
    return JSONResponse({"histogram": hist})


# ── 7b. Histogram Stretch (Germe) ────────────────────────────────────────────
@router.post("/process/histogram-stretch")
async def histogram_stretch(file: UploadFile = File(...)):
    img = _decode(await file.read())
    original_hist = compute_histogram(img)
    result = apply_histogram_stretch(img)
    processed_hist = compute_histogram(result)
    return JSONResponse({
        "result_url": _save(result, "hist_stretch"),
        "original_histogram": original_hist,
        "processed_histogram": processed_hist,
    })


# ── 7c. Histogram Equalization (Eşitleme) ────────────────────────────────────
@router.post("/process/histogram-equalize")
async def histogram_equalize(file: UploadFile = File(...)):
    img = _decode(await file.read())
    original_hist = compute_histogram(img)
    result = apply_histogram_equalization(img)
    processed_hist = compute_histogram(result)
    return JSONResponse({
        "result_url": _save(result, "hist_equalize"),
        "original_histogram": original_hist,
        "processed_histogram": processed_hist,
    })


# ── 7d. Histogram Expand (Genişletme) ────────────────────────────────────────
@router.post("/process/histogram-expand")
async def histogram_expand(
    file: UploadFile = File(...),
    a: float = Form(0.3),
    b: float = Form(0.7),
):
    img = _decode(await file.read())
    original_hist = compute_histogram(img)
    result = apply_histogram_expand(img, a=a, b=b)
    processed_hist = compute_histogram(result)
    return JSONResponse({
        "result_url": _save(result, "hist_expand"),
        "original_histogram": original_hist,
        "processed_histogram": processed_hist,
    })


# ── 8a. Add Images ───────────────────────────────────────────────────────────
@router.post("/process/add")
async def add(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = _decode(await file1.read())
    img2 = _decode(await file2.read())
    # Resize img2 to match img1 if needed (nearest-neighbor, manual)
    if img1.shape != img2.shape:
        from processing.zoom import apply_zoom
        scale_h = img1.shape[0] / img2.shape[0]
        scale_w = img1.shape[1] / img2.shape[1]
        scale = min(scale_h, scale_w)
        img2 = apply_zoom(img2, scale)
        img2 = img2[:img1.shape[0], :img1.shape[1]]
    result = add_images(img1, img2)
    return JSONResponse({"result_url": _save(result, "add")})


# ── 8b. Divide Images ────────────────────────────────────────────────────────
@router.post("/process/divide")
async def divide(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = _decode(await file1.read())
    img2 = _decode(await file2.read())
    if img1.shape != img2.shape:
        from processing.zoom import apply_zoom
        scale = img1.shape[0] / img2.shape[0]
        img2 = apply_zoom(img2, scale)
        img2 = img2[:img1.shape[0], :img1.shape[1]]
    result = divide_images(img1, img2)
    return JSONResponse({"result_url": _save(result, "divide")})


# ── 9a. Contrast Multiply ────────────────────────────────────────────────────
@router.post("/process/contrast-multiply")
async def contrast_multiply(file: UploadFile = File(...), alpha: float = Form(1.5)):
    img = _decode(await file.read())
    result = apply_contrast_multiply(img, alpha=alpha)
    return JSONResponse({"result_url": _save(result, "contrast_mul")})


# ── 9b. Contrast Log ─────────────────────────────────────────────────────────
@router.post("/process/contrast-log")
async def contrast_log(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_contrast_log(img)
    return JSONResponse({"result_url": _save(result, "contrast_log")})


# ── 10. Mean Filter ──────────────────────────────────────────────────────────
@router.post("/process/mean-filter")
async def mean_filter(file: UploadFile = File(...), size: int = Form(3)):
    img = _decode(await file.read())
    result = apply_mean_filter(img, size=size)
    return JSONResponse({"result_url": _save(result, "mean_filter")})


# ── 11. Threshold ────────────────────────────────────────────────────────────
@router.post("/process/threshold")
async def threshold(file: UploadFile = File(...), threshold_val: int = Form(128)):
    img = _decode(await file.read())
    result = apply_threshold(img, threshold=threshold_val)
    return JSONResponse({"result_url": _save(result, "threshold")})


# ── 12. Edge Detection (Prewitt) ─────────────────────────────────────────────
@router.post("/process/edge")
async def edge(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_edge_prewitt(img)
    return JSONResponse({"result_url": _save(result, "edge")})


# ── 13a. Add Noise ────────────────────────────────────────────────────────────
@router.post("/process/noise-add")
async def noise_add(file: UploadFile = File(...), amount: float = Form(0.05)):
    img = _decode(await file.read())
    result = add_salt_pepper(img, amount=amount)
    return JSONResponse({"result_url": _save(result, "noise_add")})


# ── 13b. Clean Mean ───────────────────────────────────────────────────────────
@router.post("/process/noise-mean")
async def noise_mean(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = clean_mean(img)
    return JSONResponse({"result_url": _save(result, "noise_mean")})


# ── 13c. Clean Median ─────────────────────────────────────────────────────────
@router.post("/process/noise-median")
async def noise_median(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = clean_median(img)
    return JSONResponse({"result_url": _save(result, "noise_median")})


# ── 14. Sharpening ───────────────────────────────────────────────────────────
@router.post("/process/sharpen")
async def sharpen(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_sharpening(img)
    return JSONResponse({"result_url": _save(result, "sharpen")})


# ── 15a. Dilate ───────────────────────────────────────────────────────────────
@router.post("/process/dilate")
async def dilate(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_dilate(img)
    return JSONResponse({"result_url": _save(result, "dilate")})


# ── 15b. Erode ────────────────────────────────────────────────────────────────
@router.post("/process/erode")
async def erode(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_erode(img)
    return JSONResponse({"result_url": _save(result, "erode")})


# ── 15c. Opening ─────────────────────────────────────────────────────────────
@router.post("/process/opening")
async def opening(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_opening(img)
    return JSONResponse({"result_url": _save(result, "opening")})


# ── 15d. Closing ─────────────────────────────────────────────────────────────
@router.post("/process/closing")
async def closing(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_closing(img)
    return JSONResponse({"result_url": _save(result, "closing")})
