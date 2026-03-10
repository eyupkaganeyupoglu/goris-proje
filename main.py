import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from routers import operations

# Ensure static/temp_results directory exists
os.makedirs("static/temp_results", exist_ok=True)

app = FastAPI(
    title="Image Processing App",
    description="Manual pixel-level image processing with NumPy — no high-level OpenCV functions.",
    version="1.0.0"
)

# Mount static files (CSS, JS, temp results)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template engine
templates = Jinja2Templates(directory="templates")

# Register all operation routes
app.include_router(operations.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
