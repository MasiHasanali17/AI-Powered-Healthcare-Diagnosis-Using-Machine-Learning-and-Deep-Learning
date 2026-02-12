# 🏥 AI-Driven Multi-Modal Healthcare Diagnosis System

An intelligent healthcare diagnosis platform that combines  
**Machine Learning (ML)** and **Deep Learning (DL)** to assist in disease prediction using **textual symptoms** and **medical chest X-ray images**.

---

## 📌 Project Overview

This project integrates two independent AI pipelines into a single healthcare system:

### 📝 Symptom-Based Disease Prediction (ML)
- Natural Language Processing (TF-IDF)
- Random Forest classifier
- Predicts diseases from patient-described symptoms

### 🖼️ Medical Image Diagnosis (DL)
- CNN-based ResNet architecture
- Chest X-ray image analysis
- Predicts **Normal / Pneumonia / COVID-19**
- Rejects invalid or uncertain images for safety

---

## 🧠 Technologies Used

- 🐍 **Python 3**
- ⚡ **FastAPI** – Backend REST API
- 🎨 **Streamlit** – Frontend UI
- 📊 **Scikit-learn** – Machine Learning
- 🔥 **PyTorch** – Deep Learning (CNN)
- 💾 **Joblib** – Model persistence

---

## 📂 Project Structure



AI_HEALTHCARE_DIAGNOSIS/
│
├── backend/
│ ├── main.py # FastAPI backend
│ ├── requirements.txt # Backend dependencies
│ └── init.py
│
├── frontend/
│ └── app.py # Streamlit frontend UI
│
├── ml_models/
│ ├── image_model.py # CNN inference logic
│ ├── symptom_model.py # ML inference logic
│ ├── train_image_model.py # Image model training
│ ├── train_symptom_model.py # Symptom model training
│ └── init.py
│
└── README.md


