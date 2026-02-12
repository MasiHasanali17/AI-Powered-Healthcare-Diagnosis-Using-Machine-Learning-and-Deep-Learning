

---

```md
# AI-Driven Multi-Modal Healthcare Diagnosis System

This project implements an intelligent healthcare diagnosis platform using **Machine Learning (ML)** and **Deep Learning (DL)** techniques.  
The system supports disease prediction from **textual symptoms** and **medical chest X-ray images** through a unified AI pipeline.

---

## 📌 Project Overview

The project consists of two independent AI modules:

1. **Symptom-Based Disease Prediction (Machine Learning)**
   - Uses NLP feature extraction (TF-IDF)
   - Classification using Random Forest
   - Predicts diseases from patient-described symptoms

2. **Medical Image Diagnosis (Deep Learning)**
   - Uses a CNN-based ResNet architecture
   - Analyzes chest X-ray images
   - Predicts Normal / Pneumonia / COVID-19
   - Includes uncertainty rejection for invalid images

---

## 🧠 Technologies Used

- **Python 3**
- **FastAPI** – Backend REST API
- **Streamlit** – Frontend UI
- **Scikit-learn** – ML models
- **PyTorch** – Deep Learning (CNN)
- **Joblib** – Model persistence

---

## 📂 Project Structure

```
---
AI HEALTHCARE DIAGNOSIS/
│
├── backend/
│   ├── main.py               # FastAPI backend
│   ├── requirements.txt      # Backend dependencies
│   └── **init**.py
│
├── frontend/
│   └── app.py                # Streamlit UI
│
├── ml_models/
│   ├── image_model.py        # CNN inference logic
│   ├── symptom_model.py      # ML inference logic
│   ├── train_image_model.py  # Image model training
│   ├── train_symptom_model.py# Symptom model training
│   └── **init**.py
│
└── README.md
---
````

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

Open terminal in **backend** folder:

```bash
pip install -r requirements.txt
````

---

### 2️⃣ Train Machine Learning Models

Go to **ml_models** folder:

```bash
python train_symptom_model.py
```

(Optional – only if image dataset is available)

```bash
python train_image_model.py
```

---

### 3️⃣ Start Backend Server (FastAPI)

From **backend** folder:

```bash
python -m uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### 4️⃣ Start Frontend UI (Streamlit)

From **frontend** folder:

```bash
streamlit run app.py
```

Frontend runs at:

```
http://localhost:8501
```

---

## 🧪 Sample Inputs

### Symptom Prediction (ML)

```
fever cough cold sniffles
high fever body ache chills
frequent urination thirst fatigue
chest pain pressure shortness of breath
```

### Medical Image Diagnosis (DL)

* Upload **PA-view chest X-ray images**
* Black / unrelated images are automatically rejected

---

## ⚠️ Important Notes

* This project is designed for **academic and research purposes**
* Model accuracy depends on training data size
* Confidence-based rejection ensures safe AI behavior

---

## 🎓 Academic Value

* Demonstrates real **ML + DL integration**
* Uses industry-standard architectures
* Follows reproducible AI practices
* Suitable for **Final Year / Mega Project**

---



---

## 📄 Disclaimer

This system is **not a medical device** and should not be used for real-world diagnosis.

```

---



---



Just say **NEXT** 👍
```
