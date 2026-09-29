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
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(image)
    return results