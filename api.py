import os
from fastapi import FastAPI, UploadFile, File, Form
from src.predictor import ResumePredictor
import shutil, uuid

app = FastAPI()
rp = ResumePredictor()

@app.post("/predict")
async def predict(pdf: UploadFile = File(...), jd: str = Form("")):
    os.makedirs("temp", exist_ok=True)

    path = f"temp/{uuid.uuid4()}.pdf"
    with open(path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    try:
        return rp.predict_from_pdf(path, jd)
    except Exception as e:
        return {"error": str(e)}
