import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from routers import operations

# Çıktı görselleri için klasör oluştur
os.makedirs("static/temp_results", exist_ok=True)

# FastAPI uygulamasını başlat
app = FastAPI(title="Görüntü İşleme Projesi", description="Görüntü işleme projesi kapsamında geliştirilmiştir.", version="1.0.0")

# Statik dosyaları (CSS, JS) bağla
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML şablonlarını ayarla
templates = Jinja2Templates(directory="templates")

# İşlem rotalarını ekle
app.include_router(operations.router)

# Ana sayfa
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
   return templates.TemplateResponse(request=request, name="index.html")
