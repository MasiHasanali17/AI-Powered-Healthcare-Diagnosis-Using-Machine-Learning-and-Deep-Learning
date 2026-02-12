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



---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

Open terminal inside the **backend** folder:

```bash
pip install -r requirements.txt
````

---

### 2️⃣ Train Machine Learning Models

Navigate to the **ml_models** folder:

```bash
python train_symptom_model.py
```

(Optional – only if chest X-ray dataset is available)

```bash
python train_image_model.py
```

---

### 3️⃣ Start Backend Server (FastAPI)

From the **backend** folder:

```bash
python -m uvicorn main:app --reload
```

Backend URL:

```
http://127.0.0.1:8000
```

---

### 4️⃣ Start Frontend Application (Streamlit)

From the **frontend** folder:

```bash
streamlit run app.py
```

Frontend URL:

```
http://localhost:8501
```

---

## 🧪 Sample Inputs

### 📝 Symptom Prediction (ML)

```
fever cough cold sniffles
high fever body ache chills
frequent urination thirst fatigue
chest pain pressure shortness of breath
```

### 🖼️ Medical Image Diagnosis (DL)

* Upload **PA-view chest X-ray images**
* Black, blank, or unrelated images are automatically rejected

---

## ⚠️ Important Notes

* This project is intended for **academic and research purposes**
* Prediction accuracy depends on training data size
* Confidence-based rejection ensures **safe AI behavior**

---

## 🎓 Academic Value

* Demonstrates real **ML + DL integration**
* Uses industry-standard architectures
* Follows reproducible AI practices
* Suitable for **Final Year / Mega Project**

---

## 📄 Disclaimer

⚠️ This system is **not a medical device** and should not be used for real-world clinical diagnosis.

```

---


```
