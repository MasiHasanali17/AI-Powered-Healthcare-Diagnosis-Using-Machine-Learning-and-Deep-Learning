import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Balanced medical dataset (small but valid)
data = {
    "symptoms": [
        "fever cough cold sniffles",
        "sore throat runny nose fever",

        "high fever body ache chills",
        "shivering sweating cough",

        "frequent urination thirst fatigue",
        "weight loss hunger urination",

        "chest pain pressure shortness of breath",
        "heart racing sweating pain",

        "headache nausea light sensitivity",
        "migraine throbbing head pain"
    ],
    "disease": [
        "Common Cold", "Common Cold",
        "Flu", "Flu",
        "Diabetes", "Diabetes",
        "Heart Condition", "Heart Condition",
        "Migraine", "Migraine"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["symptoms"])

# Linear SVM (good for small NLP datasets)
svm = LinearSVC(class_weight="balanced")

# FIX: cv=2 because 2 samples per class
model = CalibratedClassifierCV(svm, cv=2)
model.fit(X, df["disease"])

BASE = os.path.dirname(__file__)
joblib.dump(model, os.path.join(BASE, "symptom_model.pkl"))
joblib.dump(vectorizer, os.path.join(BASE, "vectorizer.pkl"))

print("✅ Symptom ML model trained successfully")
