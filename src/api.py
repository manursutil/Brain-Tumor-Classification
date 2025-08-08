import io
import logging
import os
import sys

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.append(os.path.abspath(".."))

from src.evaluate import evaluate_model, load_model, predict_image

WEIGHTS_PATH = "./models/resnet18_brain_mri.pt"
MAX_IMAGE_SIZE = 5 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Brain Tumor MRI Classifier",
    description="Fast-API based inference API for brain tumor MRI classification",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

model = load_model(weights_path=WEIGHTS_PATH)

def get_model():
    return model

@app.post("/predict")
async def predict(file: UploadFile = File(..., media_type="image/jpeg"), model=Depends(get_model)):  # noqa: B008
    try:
        contents = await file.read()
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_image(model, image)
        
        logger.info(f"Processed {file.filename} - result: {result}")
        return result
    except HTTPException as exc:
        raise exc
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904
    
@app.get("/metrics")
def get_metrics(model=Depends(get_model)):  # noqa: B008
    try:
        metrics = evaluate_model(model)
        return metrics
    except Exception as e:
        logger.exception("Failed to evaluate model.")
        raise HTTPException(status_code=500, detail=str(e))  # noqa: B904
    
@app.get("/health")
def health_check(model=Depends(get_model)):  # noqa: B008
    try:
        assert model is not None
        return {"status": "ok", "model_loaded": True}
    except Exception:
        return {"status": "error", "model_loaded": False}
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)