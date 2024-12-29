import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict/"

st.set_page_config(layout="wide")
st.title("Model Prediction")

image_file = st.file_uploader("Upload Image", type=["png"])

patient_id = st.number_input("Patient ID", min_value=1)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age_approx = st.number_input("Approximate Age", min_value=0, max_value=120)
anatom = {-1: 'NaN', 0: 'head/neck', 1: 'upper extremity', 2: 'lower extremity', 3: 'torso', 4: 'palms/soles', 5: 'oral/genital'}
anatom_site_general_challenge = st.selectbox("Anatomical Site", options=[-1, 0, 1, 2, 3, 4, 5], format_func=lambda x: anatom[x])

if st.button("Predict"):
    if image_file is not None:

        image_name = image_file.name
        image_bytes = image_file.read()  

        data = {"patient_id": patient_id,"sex": sex,"age_approx": age_approx,"anatom_site_general_challenge": anatom_site_general_challenge}
        files = {"file": (image_name, image_bytes, "image/png")}

        response = requests.post(API_URL, data=data, files=files)

        if response.status_code == 200:
            st.image(image_file)
            result = response.json()
            labels = {0 : 'Unknown',1: 'Nevus',2: 'Melanoma',3: 'Seborrheic Keratosis',4: 'Lentigo ',5: 'Lichenoid keratosis'}
            description = {0: 'Unknown',
            1 : "A nevus is usually dark and may be raised from the skin. Also called mole.",
            2 : "Melanoma is the most dangerous type of skin cancer; it develops from the melanin-producing cells known as melanocytes",
            3 : "Seborrheic keratosis is a common, non-cancerous (benign) skin growth that appears on the skin’s surface. It is characterized by a velvety or rough texture and can range in color from white, tan, brown, or black",
            4 : "Lentigo refers to a benign, pigmented spot on the skin with a clearly defined edge, surrounded by normal-appearing skin. It is a type of hyperplasia of melanocytes, the skin cells responsible for producing pigment",
            5 : "Lichenoid keratosis is a benign skin condition characterized by a solitary, small, raised plaque or papule, typically gray-brown in color."
            }
            st.write(f"Patient ID: {result['patient_id']}")
            st.write(f"Prediction: {labels[result['prediction']]}")
            st.write(f"Description : {description[result['prediction']]}")
        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.json())
    else:
        st.error("Please upload an image file first.")
