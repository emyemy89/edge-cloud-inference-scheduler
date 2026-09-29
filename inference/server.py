import time
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from inference.mobilenet import MobileNetInference

app = FastAPI(title="MobileNet Inference Service")

model = MobileNetInference()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    start_time = time.perf_counter()
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = model.predict(image)
    total_time = (time.perf_counter() - start_time) * 1000
    return {**result, "total_time_ms": total_time}