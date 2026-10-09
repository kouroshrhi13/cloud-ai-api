from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from model import predict_image

app = FastAPI()
app.mount("/static" , StaticFiles(directory="static") , name="static")

@app.get("/")
def home():
    return {
        "message": "Cloud AI API is running!"
    }

@app.get("/app")
def app_page():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # دریافت فایل
    contents = await file.read()

    # تبدیل فایل به تصویر PIL
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # ارسال تصویر به مدل AI
    label, confidence = predict_image(image)

    return {
        "filename": file.filename,
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }