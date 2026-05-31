import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# 1. Config Page Title
st.set_page_config(page_title="Smart Farm Analytics", layout="wide")

# 2. Load trained AI model
@st.cache_resource # Keeps model loaded in RAM for lightning-fast speeds
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# 3. Sidebar Navigation Panel
st.sidebar.title("🧭 Dashboard Menu")
app_mode = st.sidebar.radio("Go to Page:", ["🏠 Home & Live Detection", "📊 Model Validation Charts"])

# ==========================================
# PAGE 1: HOME & LIVE DETECTION + ANALYTICS
# ==========================================
if app_mode == "🏠 Home & Live Detection":
    st.title("🌱 Smart Crop & Weed Management System")
    st.write("Upload field imagery to automatically analyze weed infestation densities via deep learning.")
    
    if not model_loaded:
        st.error("⚠️ 'best.pt' file not found in this folder! Please place your trained weights file here.")
    else:
        uploaded_file = st.file_uploader("Upload an image file (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Layout columns: Left for original image, Right for AI results
            col1, col2 = st.columns(2)
            
            image = Image.open(uploaded_file)
            with col1:
                st.subheader("📷 Field Photo Uploaded")
                st.image(image, use_container_width=True)
            
            if st.button("🚀 Execute Precision AI Scanning"):
                # Run YOLO prediction
                img_array = np.array(image)
                results = model(img_array)
                
                # Fetch output and plotting arrays
                res_plotted = results[0].plot()
                classes = results[0].boxes.cls.cpu().numpy()
                
                # Run Enhancement #3: Field Analytics Calculations
                total_crops = np.sum(classes == 0)
                total_weeds = np.sum(classes == 1)
                total_plants = total_crops + total_weeds
                weed_density = (total_weeds / total_plants) * 100 if total_plants > 0 else 0
                
                with col2:
                    st.subheader("🤖 Precision AI Vision Map")
                    st.image(res_plotted, use_container_width=True)
                
                # Display Summary Dashboard Metrics
                st.markdown("---")
                st.subheader("📊 Real-Time Field Insights Summary")
                
                m1, m2, m3 = st.columns(3)
                m1.metric(label="🌾 Total Crops Identified", value=int(total_crops))
                m2.metric(label="🌿 Total Weeds Identified", value=int(total_weeds))
                m3.metric(label="⚠️ Weed Density Index", value=f"{weed_density:.2f}%")
                
                # Actionable Insights Message Engine
                if weed_density == 0 and total_plants == 0:
                    st.info("No vegetation elements detected in this quadrant sample.")
                elif weed_density > 20:
                    st.error(f"🚨 **Action Required:** Weed density is at {weed_density:.1f}%. Exceeds standard safety threshold (20%). Site-specific herbicide application is highly recommended.")
                else:
                    st.success("✅ **Field Health Stable:** Weed distribution is within localized control margins. No immediate chemical spraying required.")

# ==========================================
# PAGE 2: MODEL VALIDATION METRICS
# ==========================================
elif app_mode == "📊 Model Validation Charts":
    st.title("📈 Machine Learning Performance Verification")
    st.write("Statistical evaluation indicators and training progression histories compiled from our cloud training iterations.")
    
    # Create a local metrics folder placeholder instruction
    st.info("💡 To show your metrics here, download 'results.png' and 'confusion_matrix.png' from Google Colab and place them inside a folder named 'metrics' in your project directory.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Precision Training History Metrics")
        if os.path.exists("metrics/results.png"):
            st.image("metrics/results.png", use_container_width=True)
        else:
            st.warning("Placeholder: Add 'metrics/results.png' to populate training history trends.")
            
    with col2:
        st.subheader("🔲 Target Confusion Matrix Matrix")
        if os.path.exists("metrics/confusion_matrix.png"):
            st.image("metrics/confusion_matrix.png", use_container_width=True)
        else:
            st.warning("Placeholder: Add 'metrics/confusion_matrix.png' to populate class performance statistics.")