import joblib
import os
import numpy as np

BASE = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE, "symptom_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE, "vectorizer.pkl"))

def predict_symptoms(text: str):
    if not text or len(text.strip()) < 3:
        return {
            "disease": "Invalid Input",
            "confidence": 0.0
        }

    X = vectorizer.transform([text.lower()])
    probs = model.predict_proba(X)[0]
    idx = np.argmax(probs)

    disease = model.classes_[idx]
    confidence = float(probs[idx])

    
    if confidence < 0.45:
        return {
            "disease": "Uncertain",
            "confidence": round(confidence, 3)
        }

    return {
        "disease": disease,
        "confidence": round(confidence, 3)
    }
