import sys
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.symptom_model import predict_symptoms
from ml_models.image_model import MedicalImageAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

image_ai = MedicalImageAI()

class SymptomInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "AI Healthcare Diagnosis Online"}

@app.post("/symptoms")
def predict_symptoms_api(data: SymptomInput):
    return predict_symptoms(data.text)

@app.post("/image")
async def predict_image_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return image_ai.predict(image_bytes)