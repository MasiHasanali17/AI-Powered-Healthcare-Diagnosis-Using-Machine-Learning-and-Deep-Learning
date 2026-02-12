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

