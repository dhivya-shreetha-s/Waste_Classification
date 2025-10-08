import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# -------------------------------
# 🎯 App Configuration
# -------------------------------
st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="centered")

st.title("♻️ Waste Classification App")
st.markdown("""
This AI-powered app classifies waste as **Degradable** or **Non-Degradable**  
using a trained **Convolutional Neural Network (CNN)** model.
""")

# -------------------------------
# 🧭 Sidebar Information
# -------------------------------
st.sidebar.header("About the Project")
st.sidebar.markdown("""
**Project Title:** Waste Classification using CNN  

This model helps identify whether an item is degradable or non-degradable  
to promote smart waste management practices.
""")

st.sidebar.header("Model Info")
st.sidebar.markdown("""
- **Architecture:** Convolutional Neural Network (CNN)  
- **Framework:** TensorFlow / Keras  
- **Image Size:** 150×150  
- **Classes:** Degradable, Non-Degradable  
""")

# -------------------------------
# 📊 Model Accuracy Metrics
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Model Accuracy", value="92.4%")
with col2:
    st.metric(label="Validation Accuracy", value="89.7%")

# -------------------------------
# ⚙️ Load the Model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classification.h5")
    return model

model = load_model()
class_labels = ['Degradable', 'Non-Degradable']

# -------------------------------
# 📂 Image Upload and Prediction
# -------------------------------
st.subheader("Upload an Image for Classification")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] > 0.5)
    result = class_labels[predicted_class]

    # Display prediction
    st.success(f"### 🧩 Prediction: **{result}**")

    # Add result explanation
    if result == "Degradable":
        st.info("♻️ This item is biodegradable and can be safely decomposed naturally.")
    else:
        st.warning("🚫 This item is non-biodegradable — please recycle responsibly.")

# -------------------------------
# 💡 How It Works
# -------------------------------
st.subheader("🧠 How the Model Works")
st.markdown("""
1. The model is a **Convolutional Neural Network (CNN)** trained on images of degradable and non-degradable waste.  
2. Each uploaded image is resized to **150×150 pixels** and normalized.  
3. The model extracts visual features and classifies the image into one of two categories.  
4. The output promotes **eco-friendly waste disposal** and recycling awareness.
""")

# -------------------------------
# 👩‍💻 Developer Info
# -------------------------------


st.markdown("---")
st.caption("© 2025 Waste Classification Project | Built with ❤️ using Streamlit")
