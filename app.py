import streamlit as st
import requests

st.set_page_config(page_title="Nexus Health AI", page_icon="🏥", layout="wide")

st.title("🏥 AI Multi-Modal Healthcare Platform")
st.markdown("---")

tab1, tab2 = st.tabs(["📝 Symptom Analysis", "🖼️ Radiology AI"])

with tab1:
    st.header("NLP Symptom Prediction")
    text = st.text_area("Describe symptoms:", placeholder="e.g. I have a dry cough and fever...")
    
    if st.button("Predict Disease"):
        if text:
            res = requests.post("http://localhost:8000/symptoms", json={"text": text}).json()
            st.success(f"**Diagnosis:** {res['disease']}")
            st.info(f"**Confidence:** {res['confidence'] * 100}%")

with tab2:
    st.header("Radiology Image Diagnosis")
    img = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])
    
    if img:
        st.image(img, width=400)
        if st.button("Analyze Scan"):
            res = requests.post("http://localhost:8000/image", files={"file": img.getvalue()}).json()
            
            if res['prediction'] == "Invalid":
                st.error(" Invalid Image: Please upload a clear medical scan.")
            elif res['prediction'] == "Unknown":
                st.warning(" High Uncertainty: This image does not appear to be a Chest X-ray.")
            else:
                st.success(f"✅ Diagnosis: {res['prediction']}")
                st.write(f"Confidence: {res['confidence']}")