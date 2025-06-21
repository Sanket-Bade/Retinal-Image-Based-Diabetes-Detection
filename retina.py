import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
import mysql.connector

# Configuration
MODEL_PATH = r"C:\Users\sanke\OneDrive\Desktop\New folder (2)\resnet_model.pth"
SAVE_DIR = r"C:\Users\sanke\OneDrive\Desktop\Retina\predict"
CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Class descriptions
class_descriptions = {
    'No_DR': 'No Diabetic Retinopathy detected. Your retina appears to be healthy.',
    'Mild': 'Mild Diabetic Retinopathy. Early stages of damage to the retina, but not critical.',
    'Moderate': 'Moderate Diabetic Retinopathy. Some visible signs of damage to the retina, requires attention.',
    'Severe': 'Severe Diabetic Retinopathy. Significant damage to the retina, may require medical intervention.',
    'Proliferate_DR': 'Proliferative Diabetic Retinopathy. The most severe form of diabetic retinopathy, requiring immediate treatment.'
}

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
def load_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Transform for prediction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

# Streamlit App
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="centered")
st.title("👁️ Diabetic Retinopathy Detection")

# User Info
with st.form("user_form"):
    st.subheader("📄 Patient Information")
    name = st.text_input("Name")
    email = st.text_input("Email")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)

    medical_history = st.selectbox(
        "Do you have any medical history?",
        ["None", "Diabetes", "Hypertension", "Both"]
    )

    eye_problem = st.selectbox(
        "Do you have any eye-related symptoms?",
        ["None", "Blurriness", "Redness", "Pain", "Watery Eyes", "Other"]
    )

    captured_image = st.camera_input("📸 Capture Retina Image")

    submitted = st.form_submit_button("Submit and Predict")

if submitted:
    if captured_image is not None:
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(SAVE_DIR, f"{name}_{timestamp}.jpg")
        with open(file_path, "wb") as f:
            f.write(captured_image.getbuffer())

        # Load model and predict
        model = load_model()
        prediction = predict_image(file_path, model)

        # Get description
        description = class_descriptions.get(prediction, "No description available for this class.")

        # Display result
        st.success(f"✅ Hello {name}, Age: {age}")
        st.write(f"**Medical History**: {medical_history}")
        st.write(f"**Eye Issue**: {eye_problem}")
        st.write(f"**Prediction Result**: `{prediction}`")
        st.write(f"**Description**: {description}")

        # Optionally, save to SQL database here (if needed)
        # db_connection = mysql.connector.connect( ... )
        # cursor = db_connection.cursor()
        # cursor.execute("INSERT INTO prediction_data (name, email, age, medical_history, eye_problem, prediction) VALUES (%s, %s, %s, %s, %s, %s)", (name, email, age, medical_history, eye_problem, prediction))
        # db_connection.commit()
        # db_connection.close()

    else:
        st.warning("⚠️ Please capture an image before submitting.")
