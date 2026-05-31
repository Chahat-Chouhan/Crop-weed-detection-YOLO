import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import streamlit_authenticator as stauth

# 1. Page Config
st.set_page_config(page_title="Secure Smart Farm Analytics", layout="wide")

# ==========================================
# 🔏 AUTHENTICATION & USER MANAGEMENT SETUP
# ==========================================
# Initial pre-registered users config dictionary
# Passwords must be securely hashed in production, but the library auto-hashes new signups.
if 'config' not in st.session_state:
    st.session_state['config'] = {
        'credentials': {
            'usernames': {
                'admin': {
                    'email': 'admin@smartfarm.com',
                    'name': 'Administrator Account',
                    'password': 'password123' # Under the hood, the system authenticates this safely
                },
                'farmer1': {
                    'email': 'user@farm.com',
                    'name': 'Chahat Chouhan',
                    'password': 'cropweedyolo'
                }
            }
        },
        'cookie': {
            'expiry_days': 30,
            'key': 'smart_farm_signature_key',
            'name': 'smart_farm_auth_cookie'
        }
    }

# For simple implementation, we pre-hash existing plain text passwords for the component
for username, user_info in st.session_state['config']['credentials']['usernames'].items():
    if not user_info['password'].startswith('$2b$'): # check if not already hashed
        user_info['password'] = stauth.Hasher([user_info['password']]).generate()[0]

# Initialize Authenticator object
authenticator = stauth.Authenticate(
    st.session_state['config']['credentials'],
    st.session_state['config']['cookie']['name'],
    st.session_state['config']['cookie']['key'],
    st.session_state['config']['cookie']['expiry_days']
)

# Render the Login Widget on screen
name, authentication_status, username = authenticator.login('main', fields={'Form name': '🔐 Smart Farm AI Portal Login'})

# ==========================================
# 🚦 ROUTING LOGIC BASED ON LOGIN STATUS
# ==========================================
if authentication_status == False:
    st.error('❌ Username/password is incorrect')
    
elif authentication_status == None:
    st.warning('Please enter your credentials to access the AI Scanning Engine.')
    
    # 📝 SIGN UP NEW USER EXPANDER
    st.markdown("---")
    with st.expander("🆕 Don't have an account? Sign Up Here"):
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=False)
            if username_of_registered_user:
                st.success('🎉 User registered successfully! You can now log in above.')
                # Force updates to configuration state
                st.session_state['config']['credentials']['usernames'][username_of_registered_user] = {
                    'email': email_of_registered_user,
                    'name': name_of_registered_user,
                    'password': st.session_state['config']['credentials']['usernames'][username_of_registered_user]['password']
                }
        except Exception as e:
            st.error(f'Registration Error: {e}')

# If authentication is successful, unlock the entire main application!
elif authentication_status:
    
    # ==========================================
    # 🧠 MACHINE LEARNING CORE ENGINES LOAD
    # ==========================================
    @st.cache_resource
    def load_model():
        return YOLO('best.pt')

    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False

    # 🚪 Top Bar Welcome Area with Logout Button
    st.sidebar.markdown(f"### 👤 Welcome, **{name}**")
    authenticator.logout('Logout', 'sidebar')
    
    # Sidebar Navigation Panel
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
                col1, col2 = st.columns(2)
                image = Image.open(uploaded_file)
                
                with col1:
                    st.subheader("📷 Field Photo Uploaded")
                    st.image(image, use_container_width=True)
                
                if st.button("🚀 Execute Precision AI Scanning"):
                    img_array = np.array(image)
                    results = model(img_array)
                    
                    res_plotted = results[0].plot()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    total_crops = np.sum(classes == 0)
                    total_weeds = np.sum(classes == 1)
                    total_plants = total_crops + total_weeds
                    weed_density = (total_weeds / total_plants) * 100 if total_plants > 0 else 0
                    
                    with col2:
                        st.subheader("🤖 Precision AI Vision Map")
                        st.image(res_plotted, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📊 Real-Time Field Insights Summary")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric(label="🌾 Total Crops Identified", value=int(total_crops))
                    m2.metric(label="🌿 Total Weeds Identified", value=int(total_weeds))
                    m3.metric(label="⚠️ Weed Density Index", value=f"{weed_density:.2f}%")
                    
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
        st.write("Statistical evaluation indicators compiled from our cloud training iterations.")
        
        st.info("💡 To show your metrics here, download 'results.png' and 'confusion_matrix.png' from Google Colab and place them inside a folder named 'metrics' in your project directory.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔄 Precision Training History Metrics")
            if os.path.exists("metrics/results.png"):
                st.image("metrics/results.png", use_container_width=True)
            else:
                st.warning("Placeholder: Add 'metrics/results.png' to populate training history trends.")
                
        with col2:
            st.subheader("🔲 Target Confusion Matrix")
            if os.path.exists("metrics/confusion_matrix.png"):
                st.image("metrics/confusion_matrix.png", use_container_width=True)
            else:
                st.warning("Placeholder: Add 'metrics/confusion_matrix.png' to populate class performance statistics.")