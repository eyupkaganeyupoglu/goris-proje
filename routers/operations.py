import os
import uuid

# Temel kütüphaneler
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# İşlem fonksiyonları
from processing.grayscale import apply_grayscale
from processing.binary import apply_binary
from processing.rotation import apply_rotation
from processing.crop import apply_crop
from processing.zoom import apply_zoom
from processing.histogram import compute_histogram, apply_histogram_stretch, apply_histogram_equalization, apply_histogram_expand
from processing.arithmetic import add_images, divide_images
from processing.contrast import apply_brightness_multiply, apply_contrast_adjust
from processing.convolution import apply_mean_filter
from processing.threshold import apply_threshold
from processing.edge_detection import apply_edge_prewitt
from processing.noise import add_salt_pepper, clean_median
from processing.sharpening import apply_unsharp
from processing.morphology import apply_dilation, apply_erosion, apply_opening, apply_closing
from processing.color_space import (apply_ntsc_conversion, apply_ycbcr_conversion, apply_cmy_conversion, apply_cmyk_conversion, apply_hsi_conversion, apply_xyz_conversion, apply_lab_conversion, apply_luv_conversion)

# API yönlendiricisi ve sonuç dizini ayarları
router = APIRouter()
RESULT_DIR = "static/temp_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Bayt verisini görüntüye dönüştürür
def _decode(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="Görüntü dosyası okunamadı.")
    return img

# İşlenen görüntüyü kaydeder ve URL döner
def _save(result: np.ndarray, prefix: str) -> str:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    path = os.path.join(RESULT_DIR, filename)
    cv2.imwrite(path, result)
    return f"/{path.replace(os.sep, '/')}"

# Grayscale
@router.post("/process/grayscale")
async def grayscale(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_grayscale(img)
    return JSONResponse({"result_url": _save(result, "grayscale")})

# Binary
@router.post("/process/binary")
async def binary(file: UploadFile = File(...), threshold: int = Form(128)):
    img = _decode(await file.read())
    result = apply_binary(img, threshold=threshold)
    return JSONResponse({"result_url": _save(result, "binary")})

# Rotation
@router.post("/process/rotation")
async def rotation(file: UploadFile = File(...), angle: float = Form(45.0)):
    img = _decode(await file.read())
    result = apply_rotation(img, angle_deg=angle)
    return JSONResponse({"result_url": _save(result, "rotation")})

# Crop
@router.post("/process/crop")
async def crop(file: UploadFile = File(...), x1: int = Form(0), y1: int = Form(0), x2: int = Form(100), y2: int = Form(100)):
    img = _decode(await file.read())
    try:
        result = apply_crop(img, x1=x1, y1=y1, x2=x2, y2=y2)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    return JSONResponse({"result_url": _save(result, "crop")})

# Zoom
@router.post("/process/zoom")
async def zoom(file: UploadFile = File(...), scale: float = Form(2.0), method: str = Form("nearest")):
    img = _decode(await file.read())
    result = apply_zoom(img, scale=scale, method=method)
    return JSONResponse({"result_url": _save(result, "zoom")})

# Color Space Conversions
@router.post("/process/ntsc")
async def ntsc(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_ntsc_conversion(img)
    return JSONResponse({"result_url": _save(result, "ntsc")})

@router.post("/process/ycbcr")
async def ycbcr(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_ycbcr_conversion(img)
    return JSONResponse({"result_url": _save(result, "ycbcr")})

@router.post("/process/cmy")
async def cmy(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_cmy_conversion(img)
    return JSONResponse({"result_url": _save(result, "cmy")})

@router.post("/process/cmyk")
async def cmyk(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_cmyk_conversion(img)
    return JSONResponse({"result_url": _save(result, "cmyk")})

@router.post("/process/hsi")
async def hsi(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_hsi_conversion(img)
    return JSONResponse({"result_url": _save(result, "hsi")})

@router.post("/process/xyz")
async def xyz(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_xyz_conversion(img)
    return JSONResponse({"result_url": _save(result, "xyz")})

@router.post("/process/lab")
async def lab(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_lab_conversion(img)
    return JSONResponse({"result_url": _save(result, "lab")})

@router.post("/process/luv")
async def luv(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_luv_conversion(img)
    return JSONResponse({"result_url": _save(result, "luv")})

# Histogram Stretch (Germe)
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

# Histogram Equalization (Eşitleme)
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

# Histogram Expand (Genişletme)
@router.post("/process/histogram-expand")
async def histogram_expand(file: UploadFile = File(...), a: float = Form(0.3), b: float = Form(0.7)):
    img = _decode(await file.read())
    original_hist = compute_histogram(img)
    
    try:
        result = apply_histogram_expand(img, a=a, b=b)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
        
    processed_hist = compute_histogram(result)
    return JSONResponse({
        "result_url": _save(result, "hist_expand"),
        "original_histogram": original_hist,
        "processed_histogram": processed_hist,
    })

# Add Images
@router.post("/process/add")
async def add(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = _decode(await file1.read())
    img2 = _decode(await file2.read())
    result = add_images(img1, img2)
    return JSONResponse({"result_url": _save(result, "add")})

# Divide Images
@router.post("/process/divide")
async def divide(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = _decode(await file1.read())
    img2 = _decode(await file2.read())
    result = divide_images(img1, img2)
    return JSONResponse({"result_url": _save(result, "divide")})

# Brightness Multiply
@router.post("/process/brightness-multiply")
async def brightness_multiply(file: UploadFile = File(...), alpha: float = Form(1.5)):
    img = _decode(await file.read())
    result = apply_brightness_multiply(img, alpha=alpha)
    return JSONResponse({"result_url": _save(result, "brightness_mul")})

# Contrast Adjust
@router.post("/process/contrast-log")
async def contrast_adjust(file: UploadFile = File(...), factor: float = Form(1.5)):
    img = _decode(await file.read())
    result = apply_contrast_adjust(img, factor=factor)
    return JSONResponse({"result_url": _save(result, "contrast_adjust")})

# Mean Filter
@router.post("/process/mean-filter")
async def mean_filter(file: UploadFile = File(...), size: int = Form(3)):
    img = _decode(await file.read())
    result = apply_mean_filter(img, size=size)
    return JSONResponse({"result_url": _save(result, "mean_filter")})

# Unsharp
@router.post("/process/unsharp")
async def unsharp(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_unsharp(img)
    return JSONResponse({"result_url": _save(result, "unsharp")})

# Threshold
@router.post("/process/threshold")
async def threshold(file: UploadFile = File(...), threshold_val: int = Form(128)):
    img = _decode(await file.read())
    result = apply_threshold(img, threshold=threshold_val)
    return JSONResponse({"result_url": _save(result, "threshold")})

# Edge Detection (Prewitt)
@router.post("/process/edge")
async def edge(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_edge_prewitt(img)
    return JSONResponse({"result_url": _save(result, "edge")})

# Add Noise
@router.post("/process/noise-add")
async def noise_add(file: UploadFile = File(...), amount: float = Form(0.05)):
    img = _decode(await file.read())
    result = add_salt_pepper(img, amount=amount)
    return JSONResponse({"result_url": _save(result, "noise_add")})

# Clean Median
@router.post("/process/noise-median")
async def noise_median(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = clean_median(img)
    return JSONResponse({"result_url": _save(result, "noise_median")})

# Dilation
@router.post("/process/dilation")
async def dilation(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_dilation(img)
    return JSONResponse({"result_url": _save(result, "dilation")})

# Erosion
@router.post("/process/erosion")
async def erosion(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_erosion(img)
    return JSONResponse({"result_url": _save(result, "erosion")})

# Opening
@router.post("/process/opening")
async def opening(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_opening(img)
    return JSONResponse({"result_url": _save(result, "opening")})

# Closing
@router.post("/process/closing")
async def closing(file: UploadFile = File(...)):
    img = _decode(await file.read())
    result = apply_closing(img)
    return JSONResponse({"result_url": _save(result, "closing")})