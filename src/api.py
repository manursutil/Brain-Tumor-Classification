from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import sys
import uvicorn

sys.path.append(os.path.abspath(".."))

from src.evaluate import load_model, predict_image

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="Fast-API based inference API for brain tumor MRI classification",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

weights_path="./models/resnet18_brain_mri.pt"
model = load_model(weights_path=weights_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.save("latest_uploaded.jpg")
        result = predict_image(model, "latest_uploaded.jpg")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)